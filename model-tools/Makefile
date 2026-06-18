# mica-ai 模型工具链的顶层 Makefile
# 用法：
#   make -C model-tools help
#   make -C model-tools smoke
#   make -C model-tools download
#   make -C model-tools convert
#   make -C model-tools train-intent
#
# 单独操作某个能力：
#   make -C model-tools download-ppocr
#   make -C model-tools convert-voice
#   make -C model-tools train-intent

PYTHON ?= python
ROOT := $(shell pwd)

.PHONY: smoke
smoke: ## 离线冒烟测试（不下载任何模型）
	$(PYTHON) scripts/smoke_test.py

.PHONY: help
help: ## 显示帮助
	@echo "mica-ai 模型工具链"
	@echo ""
	@echo "常用命令："
	@echo "  make smoke          离线冒烟测试（验证骨架完整性）"
	@echo "  make download       下载所有能力的原始模型"
	@echo "  make convert        把原始模型转换为 mica-ai 可用的 ONNX 产物"
	@echo "  make train-intent   训练意图分类模型"
	@echo ""
	@echo "单能力命令："
	@echo "  make download-<cap>   cap ∈ {ppocr, tts, voice, speaker, intent}"
	@echo "  make convert-<cap>"
	@echo "  make train-intent"
	@echo ""
	@echo "环境变量："
	@echo "  PYTHON=python3.10   指定 Python 解释器"
	@echo "  MICA_MODELS_DIR=... 指定模型根目录（默认 model-tools/<cap>/model）"

CAPS := ppocr tts voice speaker intent

.PHONY: download
download: $(addprefix download-,$(CAPS)) ## 下载所有能力的模型
	@echo "✅ 全部下载完成"

.PHONY: convert
convert: $(addprefix convert-,$(CAPS)) ## 转换所有能力的模型
	@echo "✅ 全部转换完成"

# ---- 单能力 ----
define cap_template
.PHONY: download-$(1)
download-$(1): ## 下载 $(1) 的模型
	cd $(1) && $(PYTHON) download.py

.PHONY: convert-$(1)
convert-$(1): ## 转换 $(1) 的模型
	cd $(1) && $(PYTHON) convert.py
endef

$(foreach cap,$(CAPS),$(eval $(call cap_template,$(cap))))

.PHONY: train-intent
train-intent: ## 训练意图分类模型
	cd intent && $(PYTHON) train.py --config configs/base.yaml
