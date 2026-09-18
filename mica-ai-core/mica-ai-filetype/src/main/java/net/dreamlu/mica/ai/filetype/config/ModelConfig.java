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
