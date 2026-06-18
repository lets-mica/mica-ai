package net.dreamlu.mica.ai.ppocr.util;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 轻量单元测试：numpy 风格工具与框排序。
 */
class NpUtilTest {

    @Test
    void argmaxLastAxisBasic() {
        // 2 batch, 3 time, 4 classes
        float[][][] x = {
                {{0.1f, 0.2f, 0.6f, 0.1f}, {0.3f, 0.4f, 0.2f, 0.1f}, {0.5f, 0.1f, 0.3f, 0.1f}},
                {{0.0f, 0.0f, 1.0f, 0.0f}, {0.4f, 0.3f, 0.2f, 0.1f}, {0.1f, 0.9f, 0.0f, 0.0f}}
        };
        int[][] idx = NpUtil.argmaxLastAxis(x);
        assertArrayEquals(new int[]{2, 1, 0}, idx[0]);
        assertArrayEquals(new int[]{2, 0, 1}, idx[1]);
    }

    @Test
    void maxLastAxisBasic() {
        float[][][] x = {
                {{0.1f, 0.5f, 0.3f}, {0.7f, 0.2f, 0.1f}},
                {{0.4f, 0.6f, 0.0f}, {0.1f, 0.2f, 0.3f}}
        };
        float[][] m = NpUtil.maxLastAxis(x);
        assertEquals(0.5f, m[0][0], 1e-6);
        assertEquals(0.7f, m[0][1], 1e-6);
        assertEquals(0.6f, m[1][0], 1e-6);
        assertEquals(0.3f, m[1][1], 1e-6);
    }

    @Test
    void stack3DBasic() {
        List<float[][]> list = new ArrayList<>();
        list.add(new float[][]{{1, 2}, {3, 4}});
        list.add(new float[][]{{5, 6}, {7, 8}});
        float[][][] stacked = NpUtil.stack3D(list);
        assertEquals(2, stacked.length);
        assertArrayEquals(new float[]{1, 2}, stacked[0][0]);
        assertArrayEquals(new float[]{5, 6}, stacked[1][0]);
    }

    @Test
    void padRightFillsZeros() {
        float[][] x = {{1, 2, 3}, {4, 5, 6}};
        float[][] padded = NpUtil.padRight(x, 5);
        assertEquals(2, padded.length);
        assertArrayEquals(new float[]{1, 2, 3, 0, 0}, padded[0]);
        assertArrayEquals(new float[]{4, 5, 6, 0, 0}, padded[1]);
    }

    @Test
    void ceilDivBasic() {
        assertEquals(3, NpUtil.ceilDiv(10, 4));
        assertEquals(2, NpUtil.ceilDiv(8, 4));
        assertEquals(1, NpUtil.ceilDiv(1, 4));
    }

    @Test
    void clampBasic() {
        assertEquals(5, NpUtil.clamp(10, 0, 5));
        assertEquals(0, NpUtil.clamp(-1, 0, 5));
        assertEquals(3, NpUtil.clamp(3, 0, 5));
    }

    @Test
    void hwcToNchwPermutation() {
        // H=2, W=2, C=3
        float[] hwc = {
                /* (0,0) */ 1, 2, 3,
                /* (0,1) */ 4, 5, 6,
                /* (1,0) */ 7, 8, 9,
                /* (1,1) */ 10, 11, 12
        };
        float[] chw = NpUtil.hwcFlatToNchw(hwc, 2, 2, 3);
        // channel 0: 1, 4, 7, 10
        // channel 1: 2, 5, 8, 11
        // channel 2: 3, 6, 9, 12
        assertArrayEquals(new float[]{1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12}, chw, 1e-6f);
    }

    @Test
    void sortQuadBoxesByReadingOrder() {
        // 三个文本框，故意打乱
        int[][][] boxes = {
                {{100, 5}, {110, 5}, {110, 15}, {100, 15}},   // 第 2 行
                {{10, 0}, {20, 0}, {20, 10}, {10, 10}},      // 第 1 行
                {{50, 0}, {60, 0}, {60, 10}, {50, 10}}       // 第 1 行
        };
        int[][][] sorted = BoxUtil.sortQuadBoxes(boxes);
        // 排序后：第一行按 x 升序（10, 50），第二行（100）
        assertEquals(10, sorted[0][0][0]);
        assertEquals(50, sorted[1][0][0]);
        assertEquals(100, sorted[2][0][0]);
    }

    @Test
    void emptyResultsReturnEmpty() {
        assertArrayEquals(new int[0][], NpUtil.argmaxLastAxis(new float[0][][]));
        assertArrayEquals(new float[0][], NpUtil.maxLastAxis(new float[0][][]));
        assertTrue(NpUtil.empty3D().length == 0);
    }
}
