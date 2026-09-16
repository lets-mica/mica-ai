/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.recognition;

import java.util.Arrays;
import java.util.List;

/**
 * HyperLPR3 字符字典（CTC blank + 67 字符）。
 *
 * <p>对齐 Python 版 {@code hyperlpr3/common/tokenize.py}。
 */
public final class PlateDictionary {

    public static final List<String> TOKENS = Arrays.asList(
        "blank", "'", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "云", "京", "冀", "吉", "学", "宁", "川", "挂", "新", "晋", "桂",
        "民", "沪", "津", "浙", "渝", "港", "湘", "琼", "甘", "皖", "粤",
        "航", "苏", "蒙", "藏", "警", "豫", "贵", "赣", "辽", "鄂", "闽",
        "陕", "青", "鲁", "黑", "领", "使", "澳"
    );

    public static final int BLANK_INDEX = 0;

    public static int size() {
        return TOKENS.size();
    }

    private PlateDictionary() {
    }
}