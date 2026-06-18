package net.dreamlu.mica.ai.voice.engine;

import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * 轻量级 SentencePiece 分词器（仅支持 id_to_piece 查询）。
 *
 * <p>SentencePiece 的 {@code .model} 文件是一个 Protocol Buffers 序列化的
 * {@code ModelProto}，本类通过手动解析 Protobuf 二进制格式提取 piece 列表，
 * 无需引入 protobuf-java 依赖。
 *
 * <p>Protobuf schema (简化)：
 * <pre>
 * message ModelProto {
 *   repeated SentencePiece pieces = 1;  // field 1, wire type 2 (length-delimited)
 *   ...
 * }
 * message SentencePiece {
 *   string piece = 1;    // field 1, wire type 2
 *   float score = 2;     // field 2, wire type 5
 *   Type type = 3;       // field 3, wire type 0
 * }
 * </pre>
 */
@Slf4j
public final class SentencePieceTokenizer {

	/** piece 列表，index 即为 token id */
	private final List<String> pieces;

	/** 词表大小 */
	private final int pieceSize;

	public SentencePieceTokenizer(String modelPath) throws IOException {
		byte[] data = Files.readAllBytes(Path.of(modelPath));
		this.pieces = parseModelProto(data);
		this.pieceSize = pieces.size();
		log.info("SentencePiece 分词器加载完成: vocabSize={}", pieceSize);
	}

	/**
	 * 将 token id 转换为对应的 piece 字符串。
	 *
	 * @param id token id
	 * @return piece 字符串
	 */
	public String idToPiece(int id) {
		if (id < 0 || id >= pieceSize) {
			return "";
		}
		return pieces.get(id);
	}

	/**
	 * 返回词表大小。
	 */
	public int getPieceSize() {
		return pieceSize;
	}

	// ==================== Protobuf 解析 ====================

	/**
	 * 解析 ModelProto，提取所有 SentencePiece 的 piece 字段。
	 */
	private static List<String> parseModelProto(byte[] data) {
		List<String> result = new ArrayList<>();
		ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

		while (buf.hasRemaining()) {
			int tag = readVarint(buf);
			int fieldNumber = tag >>> 3;
			int wireType = tag & 0x7;

			if (fieldNumber == 1 && wireType == 2) {
				// pieces 字段：length-delimited → 嵌套 SentencePiece 消息
				int length = readVarint(buf);
				int endPos = buf.position() + length;
				String piece = parseSentencePiece(buf, endPos);
				result.add(piece);
			} else {
				skipField(buf, wireType);
			}
		}
		return result;
	}

	/**
	 * 解析单个 SentencePiece 消息，提取 piece 字段 (field 1)。
	 */
	private static String parseSentencePiece(ByteBuffer buf, int endPos) {
		String piece = "";
		while (buf.position() < endPos) {
			int tag = readVarint(buf);
			int fieldNumber = tag >>> 3;
			int wireType = tag & 0x7;

			if (fieldNumber == 1 && wireType == 2) {
				// piece 字段: string
				int len = readVarint(buf);
				byte[] bytes = new byte[len];
				buf.get(bytes);
				piece = new String(bytes, StandardCharsets.UTF_8);
			} else {
				skipField(buf, wireType);
			}
		}
		return piece;
	}

	private static int readVarint(ByteBuffer buf) {
		int result = 0;
		int shift = 0;
		while (true) {
			byte b = buf.get();
			result |= (b & 0x7F) << shift;
			if ((b & 0x80) == 0) {
				return result;
			}
			shift += 7;
			if (shift >= 35) {
				throw new RuntimeException("Varint too long");
			}
		}
	}

	private static void skipField(ByteBuffer buf, int wireType) {
		switch (wireType) {
			case 0 -> readVarint(buf); // varint
			case 1 -> buf.position(buf.position() + 8); // 64-bit
			case 2 -> { // length-delimited
				int len = readVarint(buf);
				buf.position(buf.position() + len);
			}
			case 5 -> buf.position(buf.position() + 4); // 32-bit
			default -> throw new RuntimeException("Unknown wire type: " + wireType);
		}
	}
}

