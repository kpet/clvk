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

#include <cstring>

#include "sha1.hpp"

// Implemented directly from NIST FIPS publication 180-4 (August 2015).
// https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
// Retrieved on December 22nd, 2020.

// Rotate left, defined in Section 3.2.
static inline uint32_t rotl(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

// Functions defined in Section 4.1.1.
static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ ((~x) & z);
}
static inline uint32_t parity(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}
static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

// Reverse the bytes of a word (used for little->big endian conversion).
static inline uint32_t reverse_bytes(uint32_t x) {
#ifndef WIN32
    return __builtin_bswap32(x);
#else
    return ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) |
           ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24);
#endif
}

static void sha1_process_block(cvk_sha1_hash& H, uint32_t block[16]) {
    // Constants defined in Section 4.2.1.
    const uint32_t K1 = 0x5a827999;
    const uint32_t K2 = 0x6ed9eba1;
    const uint32_t K3 = 0x8f1bbcdc;
    const uint32_t K4 = 0xca62c1d6;

    // Initialize working variables (Section 6.1.3, step 2).
    uint32_t a = H[0];
    uint32_t b = H[1];
    uint32_t c = H[2];
    uint32_t d = H[3];
    uint32_t e = H[4];

    // Round->function mapping (described in Section 4.1.1).
#define f1 ch
#define f2 parity
#define f3 maj
#define f4 parity

    // Define round functions (Section 6.1.3, step 3).
#define W(i) block[(i)&0xF]
#define round_(r, t, calc_Ws)                                                  \
    {                                                                          \
        uint32_t s = t & 0xF;                                                  \
        W(s) = calc_Ws;                                                        \
        const uint32_t T = rotl(a, 5) + f##r(b, c, d) + e + K##r + W(s);       \
        e = d;                                                                 \
        d = c;                                                                 \
        c = rotl(b, 30);                                                       \
        b = a;                                                                 \
        a = T;                                                                 \
    }
    // Used for t<16
#define round0(t) round_(1, t, reverse_bytes(W(s)))
    // Used for t>=16
#define round(r, t)                                                            \
    round_(r, t, rotl(W(s + 13) ^ W(s + 8) ^ W(s + 2) ^ W(s), 1))

    // Round 1 (with special case for first 16 rounds to swap input endianness).
    round0(0);
    round0(1);
    round0(2);
    round0(3);
    round0(4);
    round0(5);
    round0(6);
    round0(7);
    round0(8);
    round0(9);
    round0(10);
    round0(11);
    round0(12);
    round0(13);
    round0(14);
    round0(15);
    round(1, 16);
    round(1, 17);
    round(1, 18);
    round(1, 19);

    // Round 2
    round(2, 20);
    round(2, 21);
    round(2, 22);
    round(2, 23);
    round(2, 24);
    round(2, 25);
    round(2, 26);
    round(2, 27);
    round(2, 28);
    round(2, 29);
    round(2, 30);
    round(2, 31);
    round(2, 32);
    round(2, 33);
    round(2, 34);
    round(2, 35);
    round(2, 36);
    round(2, 37);
    round(2, 38);
    round(2, 39);

    // Round 3
    round(3, 40);
    round(3, 41);
    round(3, 42);
    round(3, 43);
    round(3, 44);
    round(3, 45);
    round(3, 46);
    round(3, 47);
    round(3, 48);
    round(3, 49);
    round(3, 50);
    round(3, 51);
    round(3, 52);
    round(3, 53);
    round(3, 54);
    round(3, 55);
    round(3, 56);
    round(3, 57);
    round(3, 58);
    round(3, 59);

    // Round 4
    round(4, 60);
    round(4, 61);
    round(4, 62);
    round(4, 63);
    round(4, 64);
    round(4, 65);
    round(4, 66);
    round(4, 67);
    round(4, 68);
    round(4, 69);
    round(4, 70);
    round(4, 71);
    round(4, 72);
    round(4, 73);
    round(4, 74);
    round(4, 75);
    round(4, 76);
    round(4, 77);
    round(4, 78);
    round(4, 79);

    // Compute intermediate hash for this block (Section 6.1.3, step 4).
    H[0] += a;
    H[1] += b;
    H[2] += c;
    H[3] += d;
    H[4] += e;
}

cvk_sha1_hash cvk_sha1(const uint8_t* data, uint32_t length) {
    // Initialize state (constants defined in Section 5.3.1).
    cvk_sha1_hash H;
    H[0] = 0x67452301;
    H[1] = 0xefcdab89;
    H[2] = 0x98badcfe;
    H[3] = 0x10325476;
    H[4] = 0xc3d2e1f0;

    uint32_t original_length = length;

    // Local buffer used for processing each block.
    uint32_t block[16];
    uint8_t* block_i8 = reinterpret_cast<uint8_t*>(block);

    // Process data in 512-bit blocks.
    while (length >= 64) {
        // Copy block to local buffer and process it.
        memcpy(block, data, 64);
        sha1_process_block(H, block);

        length -= 64;
        data += 64;
    }

    // Copy remaining data to local buffer.
    memcpy(block, data, length);

    // Add padding to the last block as described in Section 5.1.1.

    // Append a '1'.
    block_i8[length] = 0x80;

    // Pad rest of block with zeros.
    uint32_t padding = 64 - length - 1;
    memset(block_i8 + length + 1, 0x00, padding);

    if (padding < 8) {
        // No room for the message length.
        // Process this block as is, and then create a new one.
        sha1_process_block(H, block);
        memset(block, 0x00, 64);
    }

    // Set last 4 bytes to original length (in bits), stored big-endian.
    block[15] = reverse_bytes(original_length * 8);

    // Process final block.
    sha1_process_block(H, block);

    // Correct endianness of the result.
    H[0] = reverse_bytes(H[0]);
    H[1] = reverse_bytes(H[1]);
    H[2] = reverse_bytes(H[2]);
    H[3] = reverse_bytes(H[3]);
    H[4] = reverse_bytes(H[4]);

    return H;
}
