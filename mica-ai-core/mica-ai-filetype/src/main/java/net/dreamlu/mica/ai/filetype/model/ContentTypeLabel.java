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
package net.dreamlu.mica.ai.filetype.model;

/**
 * 特殊 / 兜底标签常量（与官方 Magika Python 版一致）。
 *
 * <ul>
 *   <li>{@link #UNKNOWN} —— 模型不在知识库 / 兜底标签</li>
 *   <li>{@link #TXT} —— 超小文本文件兜底</li>
 *   <li>{@link #EMPTY} —— 0 字节文件</li>
 *   <li>{@link #DIRECTORY} —— 目录</li>
 *   <li>{@link #SYMLINK} —— 符号链接</li>
 *   <li>{@link #UNDEFINED} —— 特殊结果（如目录、empty）下 {@code modelLabel} 字段的值</li>
 * </ul>
 */
public interface ContentTypeLabel {

    String UNKNOWN = "unknown";
    String TXT = "txt";
    String EMPTY = "empty";
    String DIRECTORY = "directory";
    String SYMLINK = "symlink";
    String UNDEFINED = "undefined";
}
