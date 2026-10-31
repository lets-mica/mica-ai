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
package net.dreamlu.mica.ai.track.autoconfigure;

import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link TrackProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>本模块刻意零依赖（不引 mica-ai-common），故参数校验用 JDK 原生
 * {@link IllegalArgumentException}，与其它能力不同。
 */
class TrackPropertiesTest {

	private static TrackProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.track", Bindable.of(TrackProperties.class))
			.orElseGet(TrackProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		TrackProperties p = new TrackProperties();
		ByteTrackConfig c = ByteTrackConfig.builder().build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getHighThreshold()).isEqualTo(c.getHighThreshold()).isEqualTo(0.6f);
		assertThat(p.getLowThreshold()).isEqualTo(c.getLowThreshold()).isEqualTo(0.1f);
		assertThat(p.getMatchThresh()).isEqualTo(c.getMatchThresh()).isEqualTo(0.8f);
		assertThat(p.getMatchLowThresh()).isEqualTo(c.getMatchLowThresh()).isEqualTo(0.5f);
		assertThat(p.getTrackBufferedThresh()).isEqualTo(c.getTrackBufferedThresh()).isEqualTo(30);
		assertThat(p.getConfirmAge()).isEqualTo(c.getConfirmAge()).isEqualTo(3);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.track.enabled", "false");
		src.put("mica.ai.track.high-threshold", "0.7");
		src.put("mica.ai.track.low-threshold", "0.2");
		src.put("mica.ai.track.match-thresh", "0.9");
		src.put("mica.ai.track.match-low-thresh", "0.6");
		src.put("mica.ai.track.track-buffered-thresh", "20");
		src.put("mica.ai.track.confirm-age", "5");

		TrackProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getHighThreshold()).isEqualTo(0.7f);
		assertThat(p.getLowThreshold()).isEqualTo(0.2f);
		assertThat(p.getMatchThresh()).isEqualTo(0.9f);
		assertThat(p.getMatchLowThresh()).isEqualTo(0.6f);
		assertThat(p.getTrackBufferedThresh()).isEqualTo(20);
		assertThat(p.getConfirmAge()).isEqualTo(5);
	}

	/**
	 * 装配一致性：{@link TrackProperties#toConfig} 必须把每个属性都装进
	 * {@link ByteTrackConfig}。
	 */
	@Test
	void toConfigShouldWireEveryProperty() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.track.high-threshold", "0.7");
		src.put("mica.ai.track.low-threshold", "0.2");
		src.put("mica.ai.track.match-thresh", "0.9");
		src.put("mica.ai.track.match-low-thresh", "0.6");
		src.put("mica.ai.track.track-buffered-thresh", "20");
		src.put("mica.ai.track.confirm-age", "5");

		ByteTrackConfig c = bind(src).toConfig();

		assertThat(c.getHighThreshold()).isEqualTo(0.7f);
		assertThat(c.getLowThreshold()).isEqualTo(0.2f);
		assertThat(c.getMatchThresh()).isEqualTo(0.9f);
		assertThat(c.getMatchLowThresh()).isEqualTo(0.6f);
		assertThat(c.getTrackBufferedThresh()).isEqualTo(20);
		assertThat(c.getConfirmAge()).isEqualTo(5);
		c.validate();
	}

	/**
	 * 非法参数必须快速失败（本模块零依赖，用 JDK 原生异常）。
	 */
	@Test
	void illegalConfigShouldFailFast() {
		ByteTrackConfig highNotGreaterThanLow = ByteTrackConfig.builder()
			.highThreshold(0.1f)
			.lowThreshold(0.6f)
			.build();
		assertThatThrownBy(highNotGreaterThanLow::validate)
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("highThreshold");

		ByteTrackConfig illegalMatchThresh = ByteTrackConfig.builder().matchThresh(0f).build();
		assertThatThrownBy(illegalMatchThresh::validate)
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("matchThresh");

		ByteTrackConfig illegalConfirmAge = ByteTrackConfig.builder().confirmAge(0).build();
		assertThatThrownBy(illegalConfirmAge::validate)
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("confirmAge");
	}
}
