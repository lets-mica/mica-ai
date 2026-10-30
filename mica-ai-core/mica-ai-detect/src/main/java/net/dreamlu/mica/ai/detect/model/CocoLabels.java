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
 * COCO 80 类英文类别名（索引即模型输出的 classId）。
 *
 * <p>NanoDet-Plus 与 YOLOX-s 均在 COCO 2017 train 上训练，类别索引一致。
 */
public final class CocoLabels {

	/**
	 * COCO 80 类顺序表
	 */
	public static final String[] COCO_80 = new String[]{
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
		"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
		"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
		"umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
		"kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
		"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
		"couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
		"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
		"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
	};

	/**
	 * COCO 80 类中文名（索引与 {@link #COCO_80} 一一对应）
	 */
	public static final String[] COCO_80_ZH = new String[]{
		"人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船",
		"交通灯", "消防栓", "停车标志", "停车计时器", "长椅", "鸟", "猫",
		"狗", "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包",
		"雨伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板", "单板滑雪", "运动球",
		"风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍",
		"瓶子", "红酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果",
		"三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子",
		"沙发", "盆栽", "床", "餐桌", "马桶", "电视", "笔记本电脑", "鼠标",
		"遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽", "冰箱",
		"书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
	};

	private CocoLabels() {
	}
}
