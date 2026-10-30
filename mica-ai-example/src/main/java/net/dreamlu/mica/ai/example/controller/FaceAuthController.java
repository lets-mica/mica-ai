/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.dreamlu.mica.ai.example.auth;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.example.pipeline.FacePipeline;
import net.dreamlu.mica.ai.example.pipeline.FaceRecognitionResult;
import net.dreamlu.mica.ai.example.repository.VectorRepository;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.autoconfigure.FaceProperties;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.LivenessResult;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.util.*;

/**
 * 自建内控级「人脸认证」最小闭环：会话挑战 → 逐帧校验 → 提交判定。
 *
 * <p>流程：
 * <ol>
 *   <li>{@code POST /face/auth/session} 下发 sessionId + 动作序列（一次性）</li>
 *   <li>{@code POST /face/auth/frame} 逐帧上传，服务端做检测 / 质量门控 / 活体 / 头部姿态</li>
 *   <li>{@code POST /face/auth/submit} 校验动作完成，用最佳帧特征与底库或参考照 1:1</li>
 * </ol>
 *
 * <p>定位是<b>自建内控级</b>（App 登录二次核验、考勤、门禁），不是银行开户级实名核身：
 * 当局不做活体检测合规认证，MiniFASNetV2 也挡不住 3D 面具与深度伪造。
 * 生产环境请把会话存储换成 Redis，并补齐设备指纹、频次限制与审计。
 */
@Slf4j
@Tag(name = "Face Auth 人脸认证", description = "会话挑战 + 多帧活体 + 1:1 比对的最小闭环")
@RestController
@RequestMapping("/face/auth")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.face", name = "enabled", havingValue = "true", matchIfMissing = true)
public class FaceAuthController {

	private static final long SESSION_TTL_MS = 120_000L;
	private static final long FRAME_INTERVAL_MS = 250L;
	private static final int MAX_FRAMES = 60;
	private static final float MIN_FACE_RATIO = 0.08f;
	private static final float MAX_FACE_RATIO = 0.90f;
	private static final double MIN_SHARPNESS = 25d;

	private final FaceAuthStore store;
	private final FacePipeline facePipeline;
	private final FaceDetector detector;
	private final FaceAligner aligner;
	private final FeatureExtractor extractor;
	private final VectorRepository vectorRepository;
	private final FaceProperties properties;

	@Operation(summary = "创建认证会话", description = "返回 sessionId 与随机顺序的动作序列，会话 120 秒过期且只能提交一次")
	@PostMapping("/session")
	public Map<String, Object> session() {
		List<AuthAction> actions = new ArrayList<>(Arrays.asList(AuthAction.values()));
		Collections.shuffle(actions, new Random());
		FaceAuthSession session = store.create(actions, SESSION_TTL_MS, MAX_FRAMES, FRAME_INTERVAL_MS);

		Map<String, Object> body = new LinkedHashMap<>();
		body.put("sessionId", session.getSessionId());
		body.put("actions", actionsBody(session));
		body.put("frameIntervalMs", session.getFrameIntervalMs());
		body.put("maxFrames", session.getMaxFrames());
		body.put("expiresInSec", SESSION_TTL_MS / 1000);
		body.put("threshold", properties.getVerify().getThreshold());
		body.put("livenessEnabled", properties.getLiveness().isEnabled());
		return body;
	}

	@Operation(summary = "上传认证帧", description = "逐帧上传；服务端返回质量、活体、姿态与动作完成度")
	@PostMapping(value = "/frame", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> frame(
		@Parameter(description = "认证会话 ID", required = true) @RequestParam("sessionId") String sessionId,
		@Parameter(description = "帧序号，从 0 开始") @RequestParam(value = "frameIndex", required = false) Integer frameIndex,
		@Parameter(description = "单帧图像", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		FaceAuthSession session = store.get(sessionId);
		if (session == null) {
			throw new ResponseStatusException(HttpStatus.GONE, "会话不存在或已过期");
		}
		if (session.getFrameCount() >= session.getMaxFrames()) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "帧数已达上限 " + session.getMaxFrames());
		}
		int frameNo = session.beginFrame();

		Map<String, Object> quality = new LinkedHashMap<>();
		double sharpness = 0d;
		int faceCount = 0;
		float faceRatio = 0f;
		Mat mat = null;
		try {
			mat = ImageUtils.byteArrayToMat(file.getBytes());
			sharpness = ImageUtils.sharpness(mat);
			FaceRecognitionResult result = facePipeline.recognize(mat);
			faceCount = result.getFaces().size();
			if (faceCount == 1) {
				FaceBox box = result.getFaces().get(0);
				faceRatio = area(box) / (float) (mat.width() * mat.height());
			}

			quality.put("sharpness", round(sharpness));
			quality.put("faceCount", faceCount);
			quality.put("faceRatio", round(faceRatio));

			if (faceCount != 1) {
				return reject(session, frameNo, faceCount == 0 ? "NO_FACE" : "MULTI_FACE",
					faceCount == 0 ? "未检测到人脸，请正对镜头" : "画面中只能有一个人脸", quality);
			}

			FaceBox box = result.getFaces().get(0);

			// 动作幅度依赖 5 点，缺点直接拒绝
			if (box.getLandmarks() == null) {
				return reject(session, frameNo, "NO_LANDMARK", "未拿到人脸关键点，请正对镜头", quality);
			}
			float[] pose = pose(box);
			if (pose == null) {
				return reject(session, frameNo, "NO_LANDMARK", "关键点距离过近，请靠近镜头", quality);
			}

		// 质量门控：清晰度 + 人脸占比
			if (sharpness < MIN_SHARPNESS) {
				return reject(session, frameNo, "BLURRY", "画面模糊，请保持稳定", quality);
			}
			if (faceRatio < MIN_FACE_RATIO) {
				return reject(session, frameNo, "FACE_TOO_SMALL", "请靠近镜头，让脸部填满取景框", quality);
			}
			if (faceRatio > MAX_FACE_RATIO) {
				return reject(session, frameNo, "FACE_TOO_LARGE", "请离镜头远一点", quality);
			}

			LivenessResult liveness = result.getLivenessMap().get(0);
			if (liveness != null && !liveness.isLive()) {
				return reject(session, frameNo, "LIVENESS_FAILED", "活体检测未通过，请用真人对准镜头",
					quality, livenessBody(liveness), poseBody(pose));
			}

			float[] feature = result.getFeatures().get(0);
			session.observePose(pose[0], pose[1]);
			session.offerFeature(feature, liveness != null ? liveness.getLiveScore() : 1f);

			Map<String, Object> body = acceptedBody(session, frameNo);
			body.put("quality", quality);
			body.put("liveness", liveness != null ? livenessBody(liveness) : null);
			body.put("pose", poseBody(pose));
			return body;
		} finally {
			ImageUtils.releaseAll(mat);
		}
	}

	@Operation(summary = "提交认证", description = "校验动作是否完成，用最佳帧特征与 personId 底库或参考照做 1:1")
	@PostMapping(value = "/submit", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> submit(
		@Parameter(description = "认证会话 ID", required = true) @RequestParam("sessionId") String sessionId,
		@Parameter(description = "底库人员 ID（未传 reference 时必填）")
		@RequestParam(value = "personId", required = false) String personId,
		@Parameter(description = "相似度阈值，默认取 mica.ai.face.verify.threshold")
		@RequestParam(value = "threshold", required = false) Float threshold,
		@Parameter(description = "参考照（可选，优先于 personId）",
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart(value = "reference", required = false) MultipartFile reference) throws IOException {
		FaceAuthSession session = store.get(sessionId);
		if (session == null) {
			throw new ResponseStatusException(HttpStatus.GONE, "会话不存在或已过期");
		}
		boolean hasReference = reference != null && !reference.isEmpty();
		if (!hasReference && (personId == null || personId.isEmpty())) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "缺少 personId 或 reference");
		}
		// 一次性消费，任何后续重放都会失败
		store.consume(sessionId);

		boolean complete = session.isComplete();
		float[] bestFeature = session.bestFeature();

		Map<String, Object> body = new LinkedHashMap<>();
		body.put("sessionId", sessionId);
		body.put("frames", session.getFrameCount());
		body.put("elapsedMs", System.currentTimeMillis() - session.getCreatedAt());
		body.put("actions", actionsBody(session));

		if (!complete) {
			body.put("passed", false);
			body.put("reason", "ACTION_INCOMPLETE");
			body.put("hint", "动作未完成，请重新开始并完成全部动作");
			return body;
		}
		if (bestFeature == null) {
			body.put("passed", false);
			body.put("reason", "LIVENESS_FAILED");
			body.put("hint", "没有可用的活体帧，请重试");
			return body;
		}

		float[] refFeature;
		if (hasReference) {
			refFeature = extractFeature(reference.getBytes());
			body.put("reference", "upload");
		} else {
			refFeature = vectorRepository.getFeature(personId);
			if (refFeature == null) {
				throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "personId 未注册: " + personId);
			}
			body.put("reference", personId);
		}

		float similarity = FeatureExtractor.compare(bestFeature, refFeature);
		float th = threshold != null ? threshold : properties.getVerify().getThreshold();
		boolean passed = similarity >= th && session.getBestLiveScore() >= livenessThreshold();

		body.put("passed", passed);
		body.put("reason", passed ? "OK" : (similarity < th ? "SIMILARITY_TOO_LOW" : "LIVENESS_FAILED"));
		body.put("similarity", round(similarity));
		body.put("threshold", th);
		body.put("livenessScore", round(session.getBestLiveScore()));
		if (!passed) {
			body.put("hint", similarity < th ? "人脸不匹配，请确认是本人操作" : "活体分不足，请重试");
		}
		log.info("人脸认证完成: personId={}, passed={}, similarity={}", personId, passed, similarity);
		return body;
	}

	private static Map<String, Object> reject(FaceAuthSession session, int frameNo,
											 String reason, String hint, Map<String, Object> quality) {
		return reject(session, frameNo, reason, hint, quality, null, null);
	}

	private static Map<String, Object> reject(FaceAuthSession session, int frameNo,
											 String reason, String hint, Map<String, Object> quality,
											 Map<String, Object> liveness, Map<String, Object> pose) {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("sessionId", session.getSessionId());
		body.put("frameIndex", frameNo - 1);
		body.put("frames", session.getFrameCount());
		body.put("maxFrames", session.getMaxFrames());
		body.put("accepted", false);
		body.put("reason", reason);
		body.put("hint", hint);
		body.put("quality", quality);
		body.put("liveness", liveness);
		body.put("pose", pose);
		body.put("actions", actionsBody(session));
		body.put("complete", session.isComplete());
		return body;
	}

	private static Map<String, Object> acceptedBody(FaceAuthSession session, int frameNo) {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("sessionId", session.getSessionId());
		body.put("frameIndex", frameNo - 1);
		body.put("frames", session.getFrameCount());
		body.put("maxFrames", session.getMaxFrames());
		body.put("accepted", true);
		body.put("reason", "OK");
		body.put("hint", session.isComplete() ? "动作已完成" : "继续按提示完成动作");
		body.put("actions", actionsBody(session));
		body.put("complete", session.isComplete());
		return body;
	}

	private static List<Map<String, Object>> actionsBody(FaceAuthSession session) {
		List<Map<String, Object>> list = new ArrayList<>();
		for (AuthAction action : session.getActions()) {
			Map<String, Object> item = new LinkedHashMap<>();
			item.put("code", action.getCode());
			item.put("label", action.getLabel());
			item.put("progress", round(session.progress(action)));
			item.put("satisfied", session.satisfied(action));
			list.add(item);
		}
		return list;
	}

	private static Map<String, Object> livenessBody(LivenessResult liveness) {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("liveScore", round(liveness.getLiveScore()));
		body.put("live", liveness.isLive());
		body.put("attackType", liveness.getAttackType());
		return body;
	}

	private static Map<String, Object> poseBody(float[] pose) {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("yaw", round(pose[0]));
		body.put("pitch", round(pose[1]));
		return body;
	}

	/**
	 * 用 5 点关键点算头部姿态比例：0 左眼 / 1 右眼 / 2 鼻尖 / 3 左嘴角 / 4 右嘴角。
	 *
	 * @return {yaw, pitch}，均按眼距归一化；关键点异常时返回 null
	 */
	private static float[] pose(FaceBox box) {
		float[][] lm = box.getLandmarks();
		float eyeDx = lm[0][0] - lm[1][0];
		float eyeDy = lm[0][1] - lm[1][1];
		float eyeDist = (float) Math.sqrt(eyeDx * eyeDx + eyeDy * eyeDy);
		if (eyeDist < 1f) {
			return null;
		}
		float eyeMidX = (lm[0][0] + lm[1][0]) / 2f;
		float eyeMidY = (lm[0][1] + lm[1][1]) / 2f;
		return new float[]{(lm[2][0] - eyeMidX) / eyeDist, (lm[2][1] - eyeMidY) / eyeDist};
	}

	private float livenessThreshold() {
		return properties.getLiveness().isEnabled() ? properties.getLiveness().getThreshold() : 0f;
	}

	/**
	 * 参考照只做检测 + 对齐 + 特征，不过活体（证件照本来就不是活体采集）。
	 */
	private float[] extractFeature(byte[] bytes) {
		Mat mat = null;
		Mat aligned = null;
		try {
			mat = ImageUtils.byteArrayToMat(bytes);
			FaceBox box = largest(detector.detect(mat));
			if (box == null) {
				throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "参考照未检测到人脸");
			}
			aligned = aligner.align(mat, box);
			return extractor.extract(aligned);
		} finally {
			ImageUtils.releaseAll(mat, aligned);
		}
	}

	private static FaceBox largest(List<FaceBox> boxes) {
		if (boxes == null || boxes.isEmpty()) {
			return null;
		}
		FaceBox best = boxes.get(0);
		for (int i = 1; i < boxes.size(); i++) {
			if (area(boxes.get(i)) > area(best)) {
				best = boxes.get(i);
			}
		}
		return best;
	}

	private static float area(FaceBox box) {
		return (box.getX2() - box.getX1()) * (box.getY2() - box.getY1());
	}

	private static double round(double value) {
		if (Double.isNaN(value) || Double.isInfinite(value)) {
			return 0d;
		}
		return Math.round(value * 1000d) / 1000d;
	}
}
