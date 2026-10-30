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
package net.dreamlu.mica.ai.track.algo;

import lombok.Getter;
import net.dreamlu.mica.ai.track.model.BBox;

/**
 * 8 维卡尔曼滤波器（中心 + 宽高 + 中心速度 + 宽高速度），标准 CV 轨迹跟踪形态。
 *
 * <p>状态向量 {@code [cx, cy, w, h, vx, vy, vw, vh]}，观测向量 {@code [cx, cy, w, h]}。
 *
 * <p>转移矩阵 F 按时间步长 1 构造：vx/vy/vw/vh 预测 cx/cy/w/h 的变化。
 *
 * <p>噪声 Q / 观测 R 为对角阵，参数与 ByteTrack 原版默认值对齐。
 */
@Getter
public class KalmanFilter {

	private static final int STATE_DIM = 8;
	private static final int MEASURE_DIM = 4;

	private final double[] x;    // 状态向量 8
	private final double[][] P;  // 协方差 8x8

	public KalmanFilter(BBox bbox) {
		this.x = new double[STATE_DIM];
		double cx = bbox.cx();
		double cy = bbox.cy();
		double w = Math.max(bbox.width(), 1e-3);
		double h = Math.max(bbox.height(), 1e-3);
		this.x[0] = cx;
		this.x[1] = cy;
		this.x[2] = w;
		this.x[3] = h;
		this.x[4] = 0;
		this.x[5] = 0;
		this.x[6] = 0;
		this.x[7] = 0;
		this.P = new double[STATE_DIM][STATE_DIM];
		for (int i = 0; i < STATE_DIM; i++) {
			P[i][i] = (i < MEASURE_DIM) ? 10.0 : 1000.0;
		}
	}

	public void predict() {
		// F: state transition (unit time step)
		// [1 0 0 0 1 0 0 0;
		//  0 1 0 0 0 1 0 0;
		//  0 0 1 0 0 0 1 0;
		//  0 0 0 1 0 0 0 1;
		//  0 0 0 0 1 0 0 0;
		//  0 0 0 0 0 1 0 0;
		//  0 0 0 0 0 0 1 0;
		//  0 0 0 0 0 0 0 1]
		x[0] += x[4];
		x[1] += x[5];
		x[2] += x[6];
		x[3] += x[7];
		// P = F * P * Ft + Q (Q diagonal)
		double[][] Q = qDiagonal();
		double[][] FP = matMul(fStateTrans(), P);
		double[][] Pnew = matAdd(matMulT(FP, fStateTrans()), Q);
		for (int i = 0; i < STATE_DIM; i++) {
			System.arraycopy(Pnew[i], 0, P[i], 0, STATE_DIM);
		}
	}

	public void update(BBox bbox) {
		double[] z = new double[]{bbox.cx(), bbox.cy(), Math.max(bbox.width(), 1e-3), Math.max(bbox.height(), 1e-3)};
		// y = z - H * x   (H 是 [I4 04])
		double[] y = new double[MEASURE_DIM];
		for (int i = 0; i < MEASURE_DIM; i++) {
			y[i] = z[i] - x[i];
		}
		// S = H * P * Ht + R
		double[][] R = rDiagonal();
		double[][] HP = new double[MEASURE_DIM][STATE_DIM];
		for (int i = 0; i < MEASURE_DIM; i++) {
			for (int j = 0; j < STATE_DIM; j++) {
				HP[i][j] = (i == j) ? P[i][j] : 0;
			}
		}
		double[][] S = new double[MEASURE_DIM][MEASURE_DIM];
		for (int i = 0; i < MEASURE_DIM; i++) {
			for (int j = 0; j < MEASURE_DIM; j++) {
				double acc = 0;
				for (int k = 0; k < STATE_DIM; k++) {
					acc += (k == i) ? P[i][k] * (k == j ? 1 : 0) : 0;
				}
				S[i][j] = acc + (i == j ? R[i][j] : 0);
			}
		}
		// 简化：S 是对角 + 小交叉近似 —— 对于目标跟踪实际 HPHt 主要也是对角主导
		// 直接用对角逆代替完整 4x4 逆
		double[] invS = new double[MEASURE_DIM];
		for (int i = 0; i < MEASURE_DIM; i++) {
			invS[i] = 1.0 / S[i][i];
		}
		// K = P * Ht * invS
		double[][] K = new double[STATE_DIM][MEASURE_DIM];
		for (int i = 0; i < STATE_DIM; i++) {
			for (int j = 0; j < MEASURE_DIM; j++) {
				K[i][j] = P[i][j] * invS[j];
			}
		}
		// x = x + K * y
		for (int i = 0; i < STATE_DIM; i++) {
			for (int j = 0; j < MEASURE_DIM; j++) {
				x[i] += K[i][j] * y[j];
			}
		}
		// P = (I - K * H) * P
		for (int i = 0; i < STATE_DIM; i++) {
			for (int j = 0; j < STATE_DIM; j++) {
				double acc = P[i][j];
				for (int k = 0; k < MEASURE_DIM; k++) {
					if (k == j) {
						acc -= K[i][k] * P[k][j];
					}
				}
				P[i][j] = acc;
			}
		}
	}

	public double predictedCx() {
		return x[0];
	}

	public double predictedCy() {
		return x[1];
	}

	public double predictedW() {
		return Math.max(1e-3, x[2]);
	}

	public double predictedH() {
		return Math.max(1e-3, x[3]);
	}

	public BBox predictedBBox() {
		double w = predictedW();
		double h = predictedH();
		double cx = predictedCx();
		double cy = predictedCy();
		return new BBox(
			(float) (cx - w * 0.5),
			(float) (cy - h * 0.5),
			(float) (cx + w * 0.5),
			(float) (cy + h * 0.5),
			0f
		);
	}

	private static double[][] fStateTrans() {
		double[][] F = new double[STATE_DIM][STATE_DIM];
		for (int i = 0; i < STATE_DIM; i++) {
			F[i][i] = 1;
		}
		for (int i = 0; i < MEASURE_DIM; i++) {
			F[i][i + MEASURE_DIM] = 1;
		}
		return F;
	}

	private static double[][] qDiagonal() {
		double[] q = new double[]{0.01, 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001, 0.0001};
		double[][] Q = new double[STATE_DIM][STATE_DIM];
		for (int i = 0; i < STATE_DIM; i++) {
			Q[i][i] = q[i];
		}
		return Q;
	}

	private static double[][] rDiagonal() {
		double[] r = new double[]{0.1, 0.1, 1.0, 1.0};
		double[][] R = new double[MEASURE_DIM][MEASURE_DIM];
		for (int i = 0; i < MEASURE_DIM; i++) {
			R[i][i] = r[i];
		}
		return R;
	}

	private static double[][] matMul(double[][] A, double[][] B) {
		int r = A.length, k = A[0].length, c = B[0].length;
		double[][] out = new double[r][c];
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				double acc = 0;
				for (int kk = 0; kk < k; kk++) {
					acc += A[i][kk] * B[kk][j];
				}
				out[i][j] = acc;
			}
		}
		return out;
	}

	private static double[][] matMulT(double[][] A, double[][] B) {
		int r = A.length, k = A[0].length, c = B.length;
		double[][] out = new double[r][c];
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				double acc = 0;
				for (int kk = 0; kk < k; kk++) {
					acc += A[i][kk] * B[j][kk];
				}
				out[i][j] = acc;
			}
		}
		return out;
	}

	private static double[][] matAdd(double[][] A, double[][] B) {
		double[][] out = new double[A.length][A[0].length];
		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A[0].length; j++) {
				out[i][j] = A[i][j] + B[i][j];
			}
		}
		return out;
	}
}
