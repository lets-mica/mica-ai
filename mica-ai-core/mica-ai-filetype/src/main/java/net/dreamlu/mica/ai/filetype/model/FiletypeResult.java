/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.filetype.PredictionMode;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class FiletypeResult {

    private String outputLabel;
    private String modelLabel;
    private float score;
    private ContentTypeInfo contentType;
    private PredictionMode mode;
    private boolean text;

    public boolean isText() {
        return text;
    }
}
