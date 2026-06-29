package net.dreamlu.mica.ai.ppocr.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import nu.pattern.OpenCV;
import org.opencv.core.Core;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.context.annotation.Bean;

/**
 * OpenCV 原生库初始化。
 *
 * <p>mica-ai-ppocr 引擎内部重度使用 {@code org.opencv.*}（{@code Imgproc} / {@code Core} / {@code Mat}）做
 * 图像预处理、文本框后处理和多边形偏移，而 openpnp/opencv 不会在 JVM 启动时自动加载 native 库，
 * 必须在 Spring 容器刷新早期显式调用 {@link OpenCV#loadShared()}，否则首次调用
 * {@code Imgproc.xxx} 时会抛 {@link UnsatisfiedLinkError}。
 *
 * <p>本类以独立的 {@code @AutoConfiguration} 形式注册，并通过
 * {@link org.springframework.boot.autoconfigure.AutoConfigureBefore @AutoConfigureBefore(PPOCRAutoConfiguration.class)}
 * 保证在 {@link PPOCRAutoConfiguration} 创建 {@code PPOcrV6Engine} 之前完成 native 加载。
 *
 * <p>启用条件：classpath 存在 {@code nu.pattern.OpenCV}（由 openpnp/opencv 传递引入）。
 * 如果用户在 {@code pom.xml} 排除了 openpnp/opencv，则本类不会注册，
 * 与 {@link PPOCRAutoConfiguration} 的 {@code @ConditionalOnClass(PPOcrV6Engine.class)} 行为保持一致。
 */
@Slf4j
@AutoConfiguration
@ConditionalOnClass(name = "nu.pattern.OpenCV")
@org.springframework.boot.autoconfigure.AutoConfigureBefore(PPOCRAutoConfiguration.class)
public class OpenCVNativeLoader {

	@Bean
	public OpenCVNativeBootstrap openCVNativeBootstrap() {
		return new OpenCVNativeBootstrap();
	}

	/**
	 * 通过工厂方法在 Bean 实例化时触发 native 加载，
	 * 确保比 {@code PPOcrV6Engine} 更早出现在容器中。
	 */
	public static class OpenCVNativeBootstrap {

		public OpenCVNativeBootstrap() {
			try {
				OpenCV.loadShared();
				log.info("[mica-ai-ppocr] OpenCV 原生库加载完成: {}", Core.VERSION);
			} catch (Throwable t) {
				log.error("[mica-ai-ppocr] OpenCV 原生库加载失败，PP-OCR Engine 将不可用", t);
			}
		}
	}
}
