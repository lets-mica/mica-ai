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

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 单条文件类型的元数据（{@code content_types_kb.min.json} 中的一行）。
 *
 * <p>字段由 Jackson 通过 {@link JsonProperty} 显式绑定 snake_case；缺失字段（除
 * {@code label} 外）会回退到安全默认值（{@code mime_type="application/octet-stream"}、
 * {@code group="other"}、{@code extensions=[]}）。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class ContentTypeInfo {

    @JsonProperty("label")
    private String label;

    @JsonProperty("mime_type")
    private String mimeType;

    @JsonProperty("group")
    private String group;

    @JsonProperty("description")
    private String description;

    @JsonProperty("extensions")
    private List<String> extensions;

    @JsonProperty("is_text")
    private boolean text;

}
