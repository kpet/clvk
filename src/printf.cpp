// Copyright 2022 The clvk authors.
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

#include <sstream>

#include "printf.hpp"

// Extract the conversion specifier from a format string
char get_fmt_conversion(std::string_view fmt) {
    auto conversionSpecPos = fmt.find_first_of("diouxXfFeEgGaAcsp");
    return fmt.at(conversionSpecPos);
}

// Read type T from given pointer
template <typename T> T read_buff(const char* data) {
    return *(reinterpret_cast<const T*>(data));
}

// Read type T from given pointer then increment the pointer
template <typename T> T read_inc_buff(char*& data) {
    T out = *(reinterpret_cast<T*>(data));
    data += sizeof(T);
    return out;
}

// Extract the optional vector flag and return a modified format string suitable
// for calling snprintf on individual vector elements
std::string get_vector_fmt(std::string fmt, int& vector_size, int& element_size,
                           std::string& remaining_fmt) {
    // Consume flags (skipping initial '%')
    auto pos = fmt.find_first_not_of(" +-#0", 1ul);
    // Consume precision and field width
    pos = fmt.find_first_not_of("123456789.", pos);

    if (fmt.at(pos) != 'v') {
        vector_size = 1;
        return std::string{fmt};
    }

    // Trim the data after the conversion specifier and store it in
    // `remaining_fmt`
    auto pos_conversion = fmt.find_first_of("diouxXfFeEgGaAcsp");
    auto fmt_specifier = fmt.substr(0, pos_conversion + 1);
    remaining_fmt = fmt.substr(pos_conversion + 1);
    fmt = fmt_specifier;

    size_t vec_length_pos_start = ++pos;
    size_t vec_length_pos_end =
        fmt.find_first_not_of("23468", vec_length_pos_start);
    auto vec_length_str = fmt.substr(vec_length_pos_start,
                                     vec_length_pos_end - vec_length_pos_start);
    int vec_length = std::atoi(vec_length_str.c_str());

    auto fmt_pre_vec_len = fmt.substr(0, vec_length_pos_start - 1);
    auto fmt_post_vec_len = fmt.substr(vec_length_pos_end, fmt.size());
    fmt = fmt_pre_vec_len + fmt_post_vec_len;

    // The length modifier is required with vectors
    if (fmt_post_vec_len.find("hh") != std::string::npos) {
        element_size = 1;
    } else if (fmt_post_vec_len.find("hl") != std::string::npos) {
        element_size = 4;
    } else if (fmt_post_vec_len.find("h") != std::string::npos) {
        element_size = 2;
    } else if (fmt_post_vec_len.find("l") != std::string::npos) {
        element_size = 8;
    }

    // If 'hl' length modifier is present, strip it as snprintf doesn't
    // understand it
    size_t hl = fmt.find("hl");
    if (hl != std::string::npos) {
        fmt.erase(hl, 2);
    }

    vector_size = vec_length;
    return fmt;
}

// Print the format part containing exactly one arg using snprintf
std::string print_part(const std::string& fmt, const char* data, size_t size) {
    // We don't know the exact size of the output string, but given we have a
    // single argument, the size of the format string plus 1024 bytes is more
    // than likely to fit everything. If it doesn't fit, just keep retrying with
    // double the output size.
    size_t out_size = fmt.size() + 1024;
    std::vector<char> out;
    out.reserve(out_size);
    out[0] = '\0';

    auto conversion = std::tolower(get_fmt_conversion(fmt));
    bool finished = false;
    while (!finished) {
        int written = 0;
        switch (conversion) {
        case 's': {
            written = snprintf(out.data(), out_size, fmt.c_str(), data);
            break;
        }
        case 'f':
        case 'e':
        case 'g':
        case 'a': {
            if (size == 4)
                written = snprintf(out.data(), out_size, fmt.c_str(),
                                   read_buff<float>(data));
            else
                written = snprintf(out.data(), out_size, fmt.c_str(),
                                   read_buff<double>(data));
            break;
        }
        default: {
            if (size == 1)
                written = snprintf(out.data(), out_size, fmt.c_str(),
                                   read_buff<uint8_t>(data));
            else if (size == 2)
                written = snprintf(out.data(), out_size, fmt.c_str(),
                                   read_buff<uint16_t>(data));
            else if (size == 4)
                written = snprintf(out.data(), out_size, fmt.c_str(),
                                   read_buff<uint32_t>(data));
            else
                written = snprintf(out.data(), out_size, fmt.c_str(),
                                   read_buff<uint64_t>(data));
            break;
        }
        }

        // Finish if the string fit in the output buffer or snprintf failed,
        // otherwise double the output buffer and try again. If snprintf failed,
        // set the output to an empty string.
        if (written < 0) {
            out[0] = '\0';
            finished = true;
        } else if (written < static_cast<long>(out_size)) {
            finished = true;
        } else {
            out_size *= 2;
            out.reserve(out_size);
        }
    }

    return std::string(out.data());
}

void process_printf(char*& data, const printf_descriptor_map_t& descs) {

    uint32_t printf_id = read_inc_buff<uint32_t>(data);
    auto& format_string = descs.at(printf_id).format_string;

    std::stringstream printf_out{};

    // Firstly print the part of the format string up to the first '%'
    size_t next_part = format_string.find_first_of('%');
    printf_out << format_string.substr(0, next_part);

    // Decompose the remaining format string into individual strings with
    // one format specifier each, handle each one individually
    size_t arg_idx = 0;
    while (next_part < format_string.size() - 1) {
        // Get the part of the format string before the next format specifier
        size_t part_start = next_part;
        size_t part_end = format_string.find_first_of('%', part_start + 1);
        auto part_fmt = format_string.substr(part_start, part_end - part_start);

        // Handle special cases
        if (part_end == part_start + 1) {
            printf_out << "%";
            next_part = part_end + 1;
            continue;
        } else if (part_end == std::string::npos &&
                   arg_idx >= descs.at(printf_id).arg_sizes.size()) {
            // If there are no remaining arguments, the rest of the format
            // should be printed verbatim
            printf_out << part_fmt;
            break;
        }

        // The size of the argument that this format part will consume
        auto& size = descs.at(printf_id).arg_sizes[arg_idx];

        // Check to see if we have a vector format specifier
        int vec_len = 0;
        int el_size = 0;
        std::string remaining_str;
        part_fmt = get_vector_fmt(part_fmt, vec_len, el_size, remaining_str);

        // Scalar argument
        if (vec_len < 2) {
            // Special case for %s
            if (get_fmt_conversion(part_fmt) == 's') {
                uint32_t string_id = read_buff<uint32_t>(data);
                printf_out << print_part(
                    part_fmt, descs.at(string_id).format_string.c_str(), size);
            } else {
                printf_out << print_part(part_fmt, data, size);
            }
            data += size;
        } else {
            // Vector argument
            auto* data_start = data;
            for (int i = 0; i < vec_len - 1; i++) {
                printf_out << print_part(part_fmt, data, size / vec_len) << ",";
                data += el_size;
            }
            printf_out << print_part(part_fmt, data, size / vec_len)
                       << remaining_str;
            data = data_start + size;
        }

        // Move to the next format part and prepare to handle the next arg
        next_part = part_end;
        arg_idx++;
    }

    printf("%s", printf_out.str().c_str());
}

void cvk_printf(cvk_mem* printf_buffer,
                const printf_descriptor_map_t& descriptors) {
    if (!printf_buffer->map()) {
        cvk_error("Could not map printf buffer");
        return;
    }
    char* data = static_cast<char*>(printf_buffer->host_va());
    auto buffer_size = printf_buffer->size();
    auto bytes_written = read_inc_buff<uint32_t>(data) * 4;
    auto* data_start = data;

    while (static_cast<size_t>(data - data_start) < bytes_written &&
           static_cast<size_t>(data - data_start) < buffer_size) {
        process_printf(data, descriptors);
    }

    printf_buffer->unmap();
}
