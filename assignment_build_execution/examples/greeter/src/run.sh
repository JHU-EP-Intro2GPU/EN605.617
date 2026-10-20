#!/usr/bin/env bash
# Asks for a name and a favorite color on standard input, then prints a
# greeting. Pass --shout to upper-case it.
set -u

shout=false
for arg in "$@"; do
    [ "$arg" = "--shout" ] && shout=true
done

printf 'Name? '
IFS= read -r name || { echo "error: no name was provided"; exit 1; }
printf 'Favorite color? '
IFS= read -r color || { echo "error: no color was provided"; exit 1; }

greeting="Hello, ${name}! Your favorite color is ${color}."
if [ "$shout" = true ]; then
    greeting="${greeting^^}"
fi
echo "$greeting"
