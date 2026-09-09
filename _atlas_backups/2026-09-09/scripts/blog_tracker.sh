#!/bin/bash
FILE=~/.hermes/blog_day.txt
if [ ! -f "$FILE" ]; then
  echo "DAY:0" > "$FILE"
fi
CURRENT=$(grep "DAY:" "$FILE" | cut -d: -f2)
NEXT=$((CURRENT + 1))
if [ "$NEXT" -gt 10 ]; then
  echo "All 10 posts complete!"
  exit 0
fi
sed -i "s/DAY:$CURRENT/DAY:$NEXT/" "$FILE"
echo "DAY:$NEXT"
