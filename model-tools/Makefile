# mica-ai 模型工具链的 Makefile
# 模型已直接入库（各能力子目录 models/，均 <50MB），无需下载脚本。
#
# 用法：
#   make -C model-tools help
#   make -C model-tools smoke

PYTHON ?= python

.PHONY: help
help: ## 显示帮助
	@echo "mica-ai 模型工具链（模型已直接入库，见各能力子目录 models/）"
	@echo ""
	@echo "命令："
	@echo "  make smoke          离线冒烟测试（校验目录与脚本完整性）"

.PHONY: smoke
smoke: ## 离线冒烟测试（不依赖网络）
	$(PYTHON) scripts/smoke_test.py
