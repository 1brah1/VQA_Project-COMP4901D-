#!/bin/bash
# scripts/jetson_fix_line_endings.sh
# Fix CRLF line endings in Jetson shell scripts if cloned on Windows

echo "[jetson] Checking for CRLF line endings in shell scripts..."

for script in scripts/jetson_*.sh; do
    if [ -f "$script" ]; then
        # Check for CRLF (carriage return + line feed)
        if grep -q $'\r' "$script"; then
            echo "[jetson] Fixing $script..."
            sed -i 's/\r$//' "$script"
        fi
    fi
done

echo "[jetson] \u2713 Line ending check complete"
