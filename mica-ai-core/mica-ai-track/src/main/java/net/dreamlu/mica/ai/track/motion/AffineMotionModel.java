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
package net.dreamlu.mica.ai.track.motion;

import net.dreamlu.mica.ai.track.model.BBox;

/**
 * 2×3 仿射变换的运动模型（纯几何、零依赖）。
 *
 * <p>变换关系：
 * <pre>
 * x' = a * x + b * y + c
 * y' = d * x + e * y + f
 * </pre>
 *
 * <p>覆盖两类来源：图像 GMC（ORB / ECC 特征匹配估出的全局运动）与 PTZ 云台角度
 * （由角度差推出的近似仿射）。
 *
 * <p><b>用法</b>：参考帧 → 当前帧的变换是逐帧复合的，所以本实例是<b>有状态</b>的 ——
 * 每帧拿到本帧的增量变换后 {@link #append(AffineMotionModel)} 累积进去：
 * <pre>{@code
 * AffineMotionModel motion = AffineMotionModel.identity();
 * // 每帧：
 * motion.append(AffineMotionModel.translation(dx, dy));   // 本帧增量
 * engine.process(detections, timestampMs);                 // 内部读 motion 的当前值
 * }</pre>
 *
 * <p><b>非线程安全</b>（内部系数可变）：一路视频流一个实例。
 *
 * <p>框的变换方式：取四角点变换后求<b>轴对齐外接矩形</b>。旋转会让外接矩形变大 ——
 * 这是几何事实，不是缺陷。
 */
public class AffineMotionModel implements MotionModel {

	private float a;
	private float b;
	private float c;
	private float d;
	private float e;
	private float f;

	/**
	 * 构造仿射运动模型。
	 *
	 * @param a x' = a*x + b*y + c 的系数 a
	 * @param b x' = a*x + b*y + c 的系数 b
	 * @param c x' = a*x + b*y + c 的系数 c
	 * @param d y' = d*x + e*y + f 的系数 d
	 * @param e y' = d*x + e*y + f 的系数 e
	 * @param f y' = d*x + e*y + f 的系数 f
	 */
	public AffineMotionModel(float a, float b, float c, float d, float e, float f) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.d = d;
		this.e = e;
		this.f = f;
	}

	/**
	 * 创建仿射运动模型。
	 *
	 * @param a x' = a*x + b*y + c 的系数 a
	 * @param b x' = a*x + b*y + c 的系数 b
	 * @param c x' = a*x + b*y + c 的系数 c
	 * @param d y' = d*x + e*y + f 的系数 d
	 * @param e y' = d*x + e*y + f 的系数 e
	 * @param f y' = d*x + e*y + f 的系数 f
	 * @return 仿射运动模型
	 */
	public static AffineMotionModel of(float a, float b, float c, float d, float e, float f) {
		return new AffineMotionModel(a, b, c, d, e, f);
	}

	/**
	 * 恒等变换（不做任何补偿）。
	 *
	 * @return 恒等运动模型
	 */
	public static AffineMotionModel identity() {
		return new AffineMotionModel(1f, 0f, 0f, 0f, 1f, 0f);
	}

	/**
	 * 纯平移。
	 *
	 * @param dx 水平位移（像素）
	 * @param dy 垂直位移（像素）
	 * @return 平移运动模型
	 */
	public static AffineMotionModel translation(float dx, float dy) {
		return new AffineMotionModel(1f, 0f, dx, 0f, 1f, dy);
	}

	/**
	 * 绕原点旋转（弧度）。
	 *
	 * <p>注意：绕<b>原点</b>（图像左上角）旋转会让框整体飞离画面 —— 相机滚转在几何上
	 * 是绕<b>主点</b>（图像中心）旋转，要那个语义请用 {@link #rotationAround(float, float, float)}。
	 *
	 * @param radians 旋转角（弧度，逆时针为正）
	 * @return 旋转运动模型
	 */
	public static AffineMotionModel rotation(float radians) {
		float cos = (float) Math.cos(radians);
		float sin = (float) Math.sin(radians);
		return new AffineMotionModel(cos, -sin, 0f, sin, cos, 0f);
	}

	/**
	 * 绕指定中心旋转（弧度）。
	 *
	 * <p>相机滚转（云台横滚、手持抖动、画面倾斜）对应的是绕<b>主点</b>的旋转，
	 * 框中心保持不动、只是外接矩形变大 —— 这才是业务上要的语义。
	 *
	 * @param radians 旋转角（弧度，逆时针为正）
	 * @param cx      旋转中心 x
	 * @param cy      旋转中心 y
	 * @return 旋转运动模型
	 */
	public static AffineMotionModel rotationAround(float radians, float cx, float cy) {
		// T(cx,cy) ∘ R ∘ T(-cx,-cy)：先平移到中心、旋转、再平移回来
		return compose(compose(translation(cx, cy), rotation(radians)), translation(-cx, -cy));
	}

	/**
	 * 复合两个变换：返回「先应用 second、再应用 first」的等效变换。
	 *
	 * @param first  后应用的变换
	 * @param second 先应用的变换
	 * @return 复合后的新实例
	 */
	public static AffineMotionModel compose(AffineMotionModel first, AffineMotionModel second) {
		if (first == null) {
			return second == null ? identity() : second.copy();
		}
		if (second == null) {
			return first.copy();
		}
		return new AffineMotionModel(
			first.a * second.a + first.b * second.d,
			first.a * second.b + first.b * second.e,
			first.a * second.c + first.b * second.f + first.c,
			first.d * second.a + first.e * second.d,
			first.d * second.b + first.e * second.e,
			first.d * second.c + first.e * second.f + first.f);
	}

	/**
	 * 累积本帧的增量变换：等价于 {@code this = this ∘ incremental}。
	 *
	 * <p><b>调用契约</b>：{@code incremental} 表示「本帧相对<b>上一帧</b>」的相机运动。
	 * 参考帧（模型状态 = 恒等的那一帧）不追加增量，从第 2 帧起每帧追加一次。
	 *
	 * @param incremental 本帧的增量变换（可为空）
	 * @return this，便于链式调用
	 */
	public AffineMotionModel append(AffineMotionModel incremental) {
		if (incremental == null) {
			return this;
		}
		AffineMotionModel composed = compose(this, incremental);
		this.a = composed.a;
		this.b = composed.b;
		this.c = composed.c;
		this.d = composed.d;
		this.e = composed.e;
		this.f = composed.f;
		return this;
	}

	/**
	 * 复制当前变换（新的独立实例）。
	 *
	 * @return 副本
	 */
	public AffineMotionModel copy() {
		return new AffineMotionModel(a, b, c, d, e, f);
	}

	/**
	 * 复位为恒等变换（换视频流 / 场景切换时用）。
	 *
	 * @return this，便于链式调用
	 */
	public AffineMotionModel reset() {
		this.a = 1f;
		this.b = 0f;
		this.c = 0f;
		this.d = 0f;
		this.e = 1f;
		this.f = 0f;
		return this;
	}

	@Override
	public BBox toCurrent(BBox referenceBox) {
		return apply(referenceBox, a, b, c, d, e, f);
	}

	@Override
	public BBox toReference(BBox currentBox) {
		float det = a * e - b * d;
		if (Math.abs(det) < 1e-9f) {
			// 退化矩阵（缩放为 0）不可逆，退化为恒等
			return currentBox;
		}
		float ia = e / det;
		float ib = -b / det;
		float id = -d / det;
		float ie = a / det;
		float ic = -(ia * c + ib * f);
		float if_ = -(id * c + ie * f);
		return apply(currentBox, ia, ib, ic, id, ie, if_);
	}

	private static BBox apply(BBox box, float a, float b, float c, float d, float e, float f) {
		if (box == null) {
			return null;
		}
		float x1 = a * box.getX1() + b * box.getY1() + c;
		float y1 = d * box.getX1() + e * box.getY1() + f;
		float x2 = a * box.getX2() + b * box.getY2() + c;
		float y2 = d * box.getX2() + e * box.getY2() + f;
		float x3 = a * box.getX1() + b * box.getY2() + c;
		float y3 = d * box.getX1() + e * box.getY2() + f;
		float x4 = a * box.getX2() + b * box.getY1() + c;
		float y4 = d * box.getX2() + e * box.getY1() + f;
		float minX = Math.min(Math.min(x1, x2), Math.min(x3, x4));
		float minY = Math.min(Math.min(y1, y2), Math.min(y3, y4));
		float maxX = Math.max(Math.max(x1, x2), Math.max(x3, x4));
		float maxY = Math.max(Math.max(y1, y2), Math.max(y3, y4));
		return new BBox(minX, minY, maxX, maxY, box.getScore(), box.getClassId());
	}

	/**
	 * 是否为恒等变换（无平移、无旋转、无缩放）。
	 *
	 * @return 恒等变换返回 true
	 */
	public boolean isIdentity() {
		return a == 1f && b == 0f && c == 0f && d == 0f && e == 1f && f == 0f;
	}

	public float getA() {
		return a;
	}

	public float getB() {
		return b;
	}

	public float getC() {
		return c;
	}

	public float getD() {
		return d;
	}

	public float getE() {
		return e;
	}

	public float getF() {
		return f;
	}

}
