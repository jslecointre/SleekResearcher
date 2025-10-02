from langchain_mcp_adapters.client import MultiServerMCPClient

from deep_research.utils import get_current_dir

# MCP server configuration for filesystem access
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
            str(get_current_dir() / "files"),  # Path to research documents
        ],
        "transport": "stdio",  # Communication via stdin/stdout
    }
}
# Global client variable - will be initialized lazily
_client = None


def get_mcp_client():
    """Get or initialize MCP client lazily to avoid issues with LangGraph Platform."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client
