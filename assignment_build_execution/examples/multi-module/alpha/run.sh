#!/usr/bin/env bash
printf 'Name? '
read -r name
echo "alpha run: args=[$*] hello $name"
