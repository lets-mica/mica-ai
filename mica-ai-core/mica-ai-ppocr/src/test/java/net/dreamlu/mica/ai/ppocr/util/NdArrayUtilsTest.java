package net.dreamlu.mica.ai.ppocr.util;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * NdArrayUtils 单元测试。
 */
class NdArrayUtilsTest {

    @Test
    void argmaxLastAxisBasic() {
        float[][][] x = {
                {{0.1f, 0.2f, 0.6f, 0.1f}, {0.3f, 0.4f, 0.2f, 0.1f}, {0.5f, 0.1f, 0.3f, 0.1f}},
                {{0.0f, 0.0f, 1.0f, 0.0f}, {0.4f, 0.3f, 0.2f, 0.1f}, {0.1f, 0.9f, 0.0f, 0.0f}}
        };
        int[][] idx = NdArrayUtils.argmaxLastAxis(x);
        assertArrayEquals(new int[]{2, 1, 0}, idx[0]);
        assertArrayEquals(new int[]{2, 0, 1}, idx[1]);
    }

    @Test
    void maxLastAxisBasic() {
        float[][][] x = {
                {{0.1f, 0.5f, 0.3f}, {0.7f, 0.2f, 0.1f}},
                {{0.4f, 0.6f, 0.0f}, {0.1f, 0.2f, 0.3f}}
        };
        float[][] m = NdArrayUtils.maxLastAxis(x);
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
        float[][][] stacked = NdArrayUtils.stack3D(list);
        assertEquals(2, stacked.length);
        assertArrayEquals(new float[]{1, 2}, stacked[0][0]);
        assertArrayEquals(new float[]{5, 6}, stacked[1][0]);
    }

    @Test
    void padRightFillsZeros() {
        float[][] x = {{1, 2, 3}, {4, 5, 6}};
        float[][] padded = NdArrayUtils.padRight(x, 5);
        assertEquals(2, padded.length);
        assertArrayEquals(new float[]{1, 2, 3, 0, 0}, padded[0]);
        assertArrayEquals(new float[]{4, 5, 6, 0, 0}, padded[1]);
    }

    @Test
    void ceilDivBasic() {
        assertEquals(3, NdArrayUtils.ceilDiv(10, 4));
        assertEquals(2, NdArrayUtils.ceilDiv(8, 4));
        assertEquals(1, NdArrayUtils.ceilDiv(1, 4));
    }

    @Test
    void clampBasic() {
        assertEquals(5, NdArrayUtils.clamp(10, 0, 5));
        assertEquals(0, NdArrayUtils.clamp(-1, 0, 5));
        assertEquals(3, NdArrayUtils.clamp(3, 0, 5));
    }

    @Test
    void hwcToNchwPermutation() {
        float[] hwc = {
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12
        };
        float[] chw = NdArrayUtils.hwcFlatToNchw(hwc, 2, 2, 3);
        assertArrayEquals(new float[]{1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12}, chw, 1e-6f);
    }

    @Test
    void sortQuadBoxesByReadingOrder() {
        int[][][] boxes = {
                {{100, 5}, {110, 5}, {110, 15}, {100, 15}},
                {{10, 0}, {20, 0}, {20, 10}, {10, 10}},
                {{50, 0}, {60, 0}, {60, 10}, {50, 10}}
        };
        int[][][] sorted = BoxUtil.sortQuadBoxes(boxes);
        assertEquals(10, sorted[0][0][0]);
        assertEquals(50, sorted[1][0][0]);
        assertEquals(100, sorted[2][0][0]);
    }

    @Test
    void emptyResultsReturnEmpty() {
        assertArrayEquals(new int[0][], NdArrayUtils.argmaxLastAxis(new float[0][][]));
        assertArrayEquals(new float[0][], NdArrayUtils.maxLastAxis(new float[0][][]));
        assertTrue(NdArrayUtils.empty3D().length == 0);
    }
}
