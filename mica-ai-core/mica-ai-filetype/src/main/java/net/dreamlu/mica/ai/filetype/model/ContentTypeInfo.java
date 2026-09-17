/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

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
