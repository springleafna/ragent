/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nageoffer.ai.ragent.core.chunk.strategy;

import com.nageoffer.ai.ragent.core.chunk.ChunkingOptions;
import com.nageoffer.ai.ragent.core.chunk.VectorChunk;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * ParagraphChunker 单元测试
 * 不启动 Spring 容器，通过测试子类直接调用 doChunk() 验证分块逻辑
 */
class ParagraphChunkerTest {

    /**
     * 通过子类暴露 protected doChunk()，同时传入 null/空依赖绕过 Embedding 调用
     */
    private static class TestableChunker extends ParagraphChunker {
        TestableChunker() {
            super(null, List.of());
        }

        List<VectorChunk> split(String text, ChunkingOptions options) {
            return doChunk(text, options);
        }
    }

    private final TestableChunker chunker = new TestableChunker();

    /**
     * 复现 Issue #4 死循环场景：
     * 当段落字符数 < overlapSize 时，原代码中 nextStart 被 clamp 至 chunkStart，
     * findParagraphIndex 返回同一 paraIndex，导致外层循环无法推进
     * 修复后必须在 3 秒内完成
     */
    @Test
    @Timeout(value = 3, unit = TimeUnit.SECONDS)
    void shouldTerminateWhenParagraphLengthShorterThanOverlap() {
        // 每段 50 字符，overlap=80 > 段落长度，触发死循环路径
        String text = "A".repeat(50) + "\n\n" + "B".repeat(50);
        ChunkingOptions options = ChunkingOptions.builder()
                .chunkSize(100)
                .overlapSize(80)
                .build();

        List<VectorChunk> chunks = chunker.split(text, options);

        assertThat(chunks).isNotEmpty();
        String allContent = chunks.stream().map(VectorChunk::getContent).collect(Collectors.joining());
        assertThat(allContent).contains("A").contains("B");
    }

    /**
     * 正常 overlap：下一个 chunk 应包含上一个 chunk 末尾内容作为上下文
     */
    @Test
    void overlapShouldIncludeTailOfPreviousChunk() {
        // 三段各 200 字符，chunkSize=300，overlap=100
        // chunk1=[para1]，之后 overlap 回退 100 字符进入 para1 尾部，产生包含 para1 尾的 overlap chunk
        String para1 = "P".repeat(200);
        String para2 = "Q".repeat(200);
        String para3 = "R".repeat(200);
        String text = para1 + "\n\n" + para2 + "\n\n" + para3;

        ChunkingOptions options = ChunkingOptions.builder()
                .chunkSize(300)
                .overlapSize(100)
                .build();

        List<VectorChunk> chunks = chunker.split(text, options);

        assertThat(chunks.size()).isGreaterThan(1);
        // overlap chunk 应含有 para1 末尾内容（全部是 'P'）
        boolean hasOverlapChunk = chunks.stream()
                .anyMatch(c -> c.getContent().startsWith("P") && !c.getContent().contains("Q"));
        assertThat(hasOverlapChunk)
                .as("应存在一个 overlap chunk，其内容来自 para1 末尾")
                .isTrue();
        // para3 的内容应出现在某个 chunk 中（不丢失末尾内容）
        boolean hasPara3 = chunks.stream().anyMatch(c -> c.getContent().contains("R"));
        assertThat(hasPara3).as("para3 的内容不应丢失").isTrue();
    }

    /**
     * 多个短段落总长度未超过 chunkSize，应合并为一个 chunk
     */
    @Test
    void shortParagraphsBelowChunkSizeShouldMergeIntoSingleChunk() {
        String text = "para1" + "\n\n" + "para2" + "\n\n" + "para3";
        ChunkingOptions options = ChunkingOptions.builder()
                .chunkSize(512)
                .overlapSize(50)
                .build();

        List<VectorChunk> chunks = chunker.split(text, options);

        assertThat(chunks).hasSize(1);
        assertThat(chunks.get(0).getContent())
                .contains("para1")
                .contains("para2")
                .contains("para3");
    }

    /**
     * 单段落超过 chunkSize（无双换行分隔）时，整段作为一个 chunk 输出
     */
    @Test
    void singleLargeParagraphShouldProduceOneChunk() {
        String text = "X".repeat(1000);
        ChunkingOptions options = ChunkingOptions.builder()
                .chunkSize(512)
                .overlapSize(128)
                .build();

        List<VectorChunk> chunks = chunker.split(text, options);

        assertThat(chunks).hasSize(1);
        assertThat(chunks.get(0).getContent()).isEqualTo(text);
    }

    /**
     * 空字符串与纯空白字符串均应返回空列表
     */
    @Test
    void emptyOrBlankTextShouldReturnEmptyList() {
        assertThat(chunker.split("", null)).isEmpty();
        assertThat(chunker.split("   ", null)).isEmpty();
    }

    /**
     * chunk index 应从 0 开始且连续递增
     */
    @Test
    void chunkIndexShouldBeSequential() {
        String text = "A".repeat(200) + "\n\n" + "B".repeat(200) + "\n\n" + "C".repeat(200);
        ChunkingOptions options = ChunkingOptions.builder()
                .chunkSize(200)
                .overlapSize(50)
                .build();

        List<VectorChunk> chunks = chunker.split(text, options);

        assertThat(chunks).isNotEmpty();
        for (int i = 0; i < chunks.size(); i++) {
            assertThat(chunks.get(i).getIndex()).isEqualTo(i);
        }
    }
}
