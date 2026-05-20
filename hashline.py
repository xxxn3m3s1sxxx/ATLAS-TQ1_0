import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'opencode-tools', 'monorepo'))
sys.exit(__import__('hashline').main())
