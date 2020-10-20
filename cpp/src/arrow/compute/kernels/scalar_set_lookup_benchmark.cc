// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "benchmark/benchmark.h"

#include "arrow/compute/api_scalar.h"
#include "arrow/compute/kernels/common.h"
#include "arrow/compute/kernels/test_util.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"
#include "arrow/util/benchmark_util.h"

namespace arrow {
namespace compute {

constexpr auto kSeed = 0x94378165;

static void SetLookupBenchmarkString(benchmark::State& state, const std::string& func_name,
                                 const int32_t valueSetLength) {
  const int64_t array_length = 1 << 20;
  const int64_t value_min_size = 0;
  const int64_t value_max_size = 32;
  const double null_probability = 0.01;
  random::RandomArrayGenerator rng(kSeed);

  auto values =
      rng.String(array_length, value_min_size, value_max_size, null_probability);
  auto value_set =
      rng.String(valueSetLength, value_min_size, value_max_size, null_probability);
  ABORT_NOT_OK(CallFunction(func_name, {values, value_set}));
  for (auto _ : state) {
    ABORT_NOT_OK(CallFunction(func_name, {values, value_set}));
  }
  state.SetItemsProcessed((state.iterations() * array_length) + (state.iterations() * valueSetLength));
  state.SetBytesProcessed((state.iterations() * values->data()->buffers[2]->size()) + (state.iterations() * value_set->data()->buffers[2]->size()));
}

template<typename Type>
static void SetLookupBenchmarkNumeric(benchmark::State& state, const std::string& func_name,
                                 const int32_t valueSetLength) {
  const int64_t array_length = 1 << 20;
  const int64_t value_min_size = 0;
  const int64_t value_max_size = 32;
  const double null_probability = 0.01;
  random::RandomArrayGenerator rng(kSeed);

  auto values =
      rng.Numeric<Type>(array_length, value_min_size, value_max_size, null_probability);
  auto value_set =
      rng.Numeric<Type>(valueSetLength, value_min_size, value_max_size, null_probability);
  ABORT_NOT_OK(CallFunction(func_name, {values, value_set}));
  for (auto _ : state) {
    ABORT_NOT_OK(CallFunction(func_name, {values, value_set}));
  }
  state.SetItemsProcessed((state.iterations() * array_length) + (state.iterations() * valueSetLength));
  state.SetBytesProcessed((state.iterations() * values->data()->buffers[1]->size()) + (state.iterations() * value_set->data()->buffers[1]->size()));
}

static void IndexInStringSmall(benchmark::State& state) {
    SetLookupBenchmarkString(state, "index_in_meta_binary", state.range(0));
}

static void IsInStringSmall(benchmark::State& state) {
    SetLookupBenchmarkString(state, "is_in_meta_binary", state.range(0));
}

static void IndexInStringLarge(benchmark::State& state) {
    SetLookupBenchmarkString(state, "index_in_meta_binary", 1 << 20);
}

static void IsInStringLarge(benchmark::State& state) {
    SetLookupBenchmarkString(state, "is_in_meta_binary", 1 << 20);
}

static void IndexInInt8Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int8Type>(state, "index_in_meta_binary", state.range(0));
}

static void IndexInInt16Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int16Type>(state, "index_in_meta_binary", state.range(0));
}

static void IndexInInt32Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int32Type>(state, "index_in_meta_binary", state.range(0));
}

static void IndexInInt64Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int64Type>(state, "index_in_meta_binary", state.range(0));
}

static void IsInInt8Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int8Type>(state, "is_in_meta_binary", state.range(0));
}

static void IsInInt16Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int16Type>(state, "is_in_meta_binary", state.range(0));
}

static void IsInInt32Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int32Type>(state, "is_in_meta_binary", state.range(0));
}

static void IsInInt64Small(benchmark::State& state) {
    SetLookupBenchmarkNumeric<Int64Type>(state, "is_in_meta_binary", state.range(0));
}

BENCHMARK(IndexInStringSmall)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IsInStringSmall)->RangeMultiplier(2)->Range(2, 256);

BENCHMARK(IndexInStringLarge);
BENCHMARK(IsInStringLarge);

BENCHMARK(IndexInInt8Small)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IndexInInt16Small)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IndexInInt32Small)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IndexInInt64Small)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IsInInt8Small)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IsInInt16Small)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IsInInt32Small)->RangeMultiplier(2)->Range(2, 256);
BENCHMARK(IsInInt64Small)->RangeMultiplier(2)->Range(2, 256);

}  // namespace compute
}  // namespace arrow
