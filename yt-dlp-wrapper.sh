#!/bin/bash
# yt-dlp wrapper script that automatically adds Firefox cookies for authentication
# This wrapper ensures yt-dlp always uses Firefox cookies to bypass YouTube authentication

# Call the original yt-dlp with Firefox cookies and pass through all arguments
exec /usr/local/bin/yt-dlp-original --cookies-from-browser firefox "$@"