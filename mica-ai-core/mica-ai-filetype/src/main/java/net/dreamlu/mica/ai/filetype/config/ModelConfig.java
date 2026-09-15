/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

@Data
@NoArgsConstructor
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    @JsonProperty("beg_size")
    private int begSize;

    @JsonProperty("mid_size")
    private int midSize;

    @JsonProperty("end_size")
    private int endSize;

    @JsonProperty("use_inputs_at_offsets")
    private boolean useInputsAtOffsets;

    @JsonProperty("medium_confidence_threshold")
    private float mediumConfidenceThreshold;

    @JsonProperty("min_file_size_for_dl")
    private int minFileSizeForDl;

    @JsonProperty("padding_token")
    private int paddingToken;

    @JsonProperty("block_size")
    private int blockSize;

    @JsonProperty("target_labels_space")
    private List<String> targetLabelsSpace;

    @JsonProperty("thresholds")
    private Map<String, Float> thresholds;

    @JsonProperty("overwrite_map")
    private Map<String, String> overwriteMap;

    public int featuresSize() {
        return begSize + endSize;
    }
}
