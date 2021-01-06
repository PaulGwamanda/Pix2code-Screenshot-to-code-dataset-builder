#!/usr/bin/env bash
// Bash script to create blank new files:
// Go to websites.txt and give the filename with extension that you want to create, ie website-1.gui
// Then run script below:

for /f "delims=" %F in (websites.txt) do copy nul "%F"