import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

public class RunHoubbTest {
	public static void main(String[] args) throws Exception {
		Class<?> testClass = Class.forName("net.dreamlu.mica.ai.tts.g2p.HoubbPinyinG2PTest");
		Object test = testClass.getDeclaredConstructor().newInstance();

		List<String> results = new ArrayList<>();
		results.add("=== HoubbPinyinG2P 单元测试 ===\n");

		// 1. testConvertReturnsBopomofo
		try {
			Method m = testClass.getMethod("testConvertReturnsBopomofo");
			m.invoke(test);
			results.add("[OK] testConvertReturnsBopomofo");
		} catch (Throwable t) {
			results.add("[FAIL] testConvertReturnsBopomofo: " + rootCause(t));
		}

		// 2. testConvertEmpty
		try {
			Method m = testClass.getMethod("testConvertEmpty");
			m.invoke(test);
			results.add("[OK] testConvertEmpty");
		} catch (Throwable t) {
			results.add("[FAIL] testConvertEmpty: " + rootCause(t));
		}

		// 3. testConvertPureEnglish
		try {
			Method m = testClass.getMethod("testConvertPureEnglish");
			m.invoke(test);
			results.add("[OK] testConvertPureEnglish");
		} catch (Throwable t) {
			results.add("[FAIL] testConvertPureEnglish: " + rootCause(t));
		}

		// 4. testConvertPolyphone
		try {
			Method m = testClass.getMethod("testConvertPolyphone");
			m.invoke(test);
			results.add("[OK] testConvertPolyphone");
		} catch (Throwable t) {
			results.add("[FAIL] testConvertPolyphone: " + rootCause(t));
		}

		// 5. testConvertDigits
		try {
			Method m = testClass.getMethod("testConvertDigits");
			m.invoke(test);
			results.add("[OK] testConvertDigits");
		} catch (Throwable t) {
			results.add("[FAIL] testConvertDigits: " + rootCause(t));
		}

		// 顺便看看实际输出
		Class<?> g2pClass = Class.forName("net.dreamlu.mica.ai.tts.g2p.HoubbPinyinG2P");
		Object g2p = g2pClass.getDeclaredConstructor().newInstance();
		Method convert = g2pClass.getMethod("convert", String.class);
		results.add("\n=== 实际输出 ===");
		results.add("Hello World       -> [" + convert.invoke(g2p, "Hello World") + "]");
		results.add("mica-ai           -> [" + convert.invoke(g2p, "mica-ai") + "]");
		results.add("你好，欢迎使用 mica-ai。 -> [" + convert.invoke(g2p, "你好，欢迎使用 mica-ai。") + "]");
		results.add("Hello mica-ai World -> [" + convert.invoke(g2p, "Hello mica-ai World") + "]");

		for (String r : results) {
			System.out.println(r);
		}
	}

	private static String rootCause(Throwable t) {
		Throwable cur = t;
		while (cur.getCause() != null && cur.getCause() != cur) {
			cur = cur.getCause();
		}
		return cur.getClass().getSimpleName() + ": " + cur.getMessage();
	}
}
