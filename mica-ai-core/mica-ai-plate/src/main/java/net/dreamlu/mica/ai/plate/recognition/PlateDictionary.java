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
package net.dreamlu.mica.ai.plate.recognition;

import java.util.Arrays;
import java.util.List;

/**
 * HyperLPR3 字符字典（CTC blank + 71 token，对齐 Python 版
 * {@code hyperlpr3/common/tokenize.py}）。
 *
 * <p>下标 0 为 CTC blank；其余 1-9 数字、10-35 英文大写（跳过 I / O）、
 * 36-70 为 31 个省份简称 + 学 / 警 / 使 / 领 / 港 / 澳。
 *
 * <p>线程安全（不可变常量列表）。
 */
public final class PlateDictionary {

    /** 字符 token 列表，下标 0 为 CTC blank */
    public static final List<String> TOKENS = Arrays.asList(
        "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂",
        "民", "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤",
        "航", "苏", "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽",
        "陕", "青", "鲁", "黑", "领", "使", "澳"
    );

    /** CTC blank 在 token 列表中的下标 */
    public static final int BLANK_INDEX = 0;

    /**
     * 获取字典 token 总数。
     *
     * @return token 数量
     */
    public static int size() {
        return TOKENS.size();
    }

    private PlateDictionary() {
    }
}