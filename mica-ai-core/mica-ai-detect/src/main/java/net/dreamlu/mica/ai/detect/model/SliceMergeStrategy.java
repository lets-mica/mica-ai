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
package net.dreamlu.mica.ai.detect.model;

/**
 * 切片推理的跨片合并策略。
 */
public enum SliceMergeStrategy {

	/**
	 * 跨片 IoU 合并（默认）：把各切片的结果汇总后做一次 NMS，
	 * 同一物体被多个切片看到时只保留得分最高的框。
	 *
	 * <p>类别语义与模型族一致：NanoDet-Plus 类别无关，YOLOX-s 按类别互斥。
	 */
	IOU_MERGE,

	/**
	 * 不合并：保留各切片原始框，由调用方自行做跨片去重。
	 */
	KEEP_ALL

}
