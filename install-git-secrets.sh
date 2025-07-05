#!/bin/bash

git secrets --install

git secrets --add 'sk_live_[0-9a-zA-Z]{24}'
git secrets --add 'sk_test_[0-9a-zA-Z]{24}'
git secrets --add --literal 'GOOGLE_API_KEY'
git secrets --add --literal 'LANGFUSE_SECRET_KEY'
git secrets --add --literal 'Authorization:'
git secrets --add 'AIza[0-9A-Za-z_\-]{35}'

echo "git-secrets installed with Orki-safe patterns"
