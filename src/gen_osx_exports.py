#!/usr/bin/env python3
import re
import sys

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    symbols = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Match lines ending with semicolon that look like symbol names
            # This regex avoids keywords like "global:" or "local:" because of the strict match
            match = re.match(r'^([a-zA-Z0-9_]+);$', line)
            if match:
                symbol = match.group(1)
                symbols.append(symbol)

    with open(output_file, 'w') as f:
        for symbol in symbols:
            f.write(f'_{symbol}\n')

if __name__ == "__main__":
    main()

