// Copyright 2020 The clvk authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstdint>

constexpr unsigned SHA1_DIGEST_NUM_WORDS = 5;
constexpr unsigned SHA1_DIGEST_NUM_BYTES = 20;

using cvk_sha1_hash = std::array<uint32_t, 5>;

// Compute the SHA-1 hash for `length` bytes of `data`.
cvk_sha1_hash cvk_sha1(const void* data, uint32_t length);
