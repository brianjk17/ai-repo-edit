from typing import Dict, Any, List, Annotated
import git
from pathlib import Path
import os
from github import Github
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import operator
from dotenv import load_dotenv
import slack
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import threading
import time
import traceback

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = init_chat_model("openai:gpt-4o-mini")

# Initialize Slack app
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
app_token = os.environ['SLACK_APP_TOKEN']
user_token = os.environ['SLACK_OAUTH_TOKEN']

client = slack.WebClient(token=user_token)
app = App(token=user_token)

def replace_value(old: str, new: str) -> str:
    return new

def merge_dicts(old: dict, new: dict) -> dict:
    return {**old, **new}

class State(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]  
    repo_path: Annotated[str, replace_value] 
    repo_url: Annotated[str, replace_value]  
    instruction: Annotated[str, replace_value] 
    code: Annotated[str, replace_value]  
    modification: Annotated[str, replace_value]
    branch: Annotated[str, replace_value]  
    changes: Annotated[dict, merge_dicts] 
    base_branch_name: Annotated[str, replace_value]
    slack_channel: Annotated[str, replace_value]
    slack_user: Annotated[str, replace_value]

def send_message_to_channel(channel, message):
    """Send message to Slack channel"""
    try:
        client.chat_postMessage(channel=channel, text=message)
    except Exception as e:
        print(f"Error sending message to Slack: {e}")

def clone_repo(repo_url: str, clone_path: str = "./repo") -> git.Repo:
    if not repo_url:
        raise ValueError("Repository URL cannot be empty")
    repo_path = Path(clone_path)
    
    # Delete existing repo directory if it exists
    if repo_path.exists():
        print(f"Deleting existing repository at {repo_path}")
        import shutil
        shutil.rmtree(repo_path)
    
    print(f"Cloning repository to {repo_path}")
    return git.Repo.clone_from(repo_url, repo_path)

def clone_node(state: Dict[str, Any]) -> Dict[str, Any]:
    repo_url = state.get("repo_url")
    channel = state.get("slack_channel")
    
    send_message_to_channel(channel, f"🔄 Cloning repository: {repo_url}")
    
    print(f"Cloning repository: {repo_url}")
    if not repo_url:
        raise ValueError("Repository URL is required")
    
    # Clone the repository and get the path
    repo = clone_repo(repo_url)
    repo_path = str(repo.working_dir)
    print(f"Cloned repository to: {repo_path}")
    
    send_message_to_channel(channel, f"✅ Successfully cloned repository to {repo_path}")
    
    new_messages = [{"role": "system", "content": f"Cloned repository from {repo_url} to {repo_path}"}]
    
    return {
        "repo_path": repo_path,
        "messages": new_messages
    }

def read_repo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    repo_path = state.get("repo_path")
    channel = state.get("slack_channel")
    
    if not repo_path or not isinstance(repo_path, str):
        raise ValueError(f"Invalid repository path: {repo_path}")
    
    send_message_to_channel(channel, f"📖 Reading repository files...")
    
    print(f"Reading repository from path: {repo_path}")
    code = ""
    path = Path(repo_path)
    
    if not path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    file_count = 0
    for p in path.rglob("*"):
        if p.is_file() and not any(part.startswith('.') for part in p.parts):
            try:
                code += f"\n# File: {p}\n" + p.read_text()
                file_count += 1
            except Exception as e:
                code += f"\n# Error reading {p}: {e}"
    
    send_message_to_channel(channel, f"✅ Read {file_count} files ({len(code)} characters total)")
    
    new_messages = [{"role": "system", "content": f"Read {len(code)} characters of code from repository"}]
    
    return {
        "code": code,
        "messages": new_messages
    }

def apply_change_node(state: Dict[str, Any]) -> Dict[str, Any]:
    code = state.get("code", "")
    instruction = state.get("instruction", "")
    channel = state.get("slack_channel")
    
    if instruction.lower() == "implement":
        send_message_to_channel(channel, "🚀 Implementing the changes...")
        return state
        
    if not code:
        send_message_to_channel(channel, "❌ Error: No code content found")
        return state
    
    send_message_to_channel(channel, f"🤖 Processing instruction: {instruction}")
    
    # Get conversation history for context
    messages = state.get("messages", [])
    
    system_messages = [SystemMessage(content="You're a helpful AI software engineer. Please provide the complete modified code for each file that needs to be changed. Consider the conversation history to understand the context and build upon previous changes. Format your response as:\n\n# File: path/to/file\n```\nfile contents\n```\n\n# File: path/to/another/file\n```\nfile contents\n```")]
    
    # Convert messages to LangChain message objects
    conversation_history = []
    for msg in messages:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                conversation_history.append(HumanMessage(content=f"User instruction: {msg['content']}"))
            elif msg["role"] == "assistant":
                conversation_history.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                conversation_history.append(SystemMessage(content=msg["content"]))
        elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
            conversation_history.append(msg)
    
    # Build the prompt with conversation context
    prompt_messages = system_messages + conversation_history + [
        HumanMessage(content=f"Current codebase:\n\n{code}"),
        HumanMessage(content=f"Latest instruction: {instruction}")
    ]
    
    response = llm.invoke(prompt_messages)
    
    # Parse the response to get file changes
    changes = {}
    current_file = None
    current_content = []
    
    for line in response.content.split('\n'):
        if line.startswith('# File: '):
            if current_file and current_content:
                changes[current_file] = '\n'.join(current_content)
            current_file = line[8:].strip()
            current_content = []
        elif line.startswith('```'):
            continue
        elif current_file:
            current_content.append(line)
    
    if current_file and current_content:
        changes[current_file] = '\n'.join(current_content)
    
    # Send the proposed changes to Slack
    if not changes:
        send_message_to_channel(channel, "❌ No changes were proposed. The instruction might need to be more specific.")
        # Add AI response to conversation
        ai_message = {"role": "assistant", "content": "No changes were proposed. Please provide more specific instructions."}
        state["messages"].append(ai_message)
    else:
        change_summary = f"✅ Proposed changes to {len(changes)} files:\n"
        for file_path in changes.keys():
            change_summary += f"• {file_path}\n"
        
        send_message_to_channel(channel, change_summary)
        send_message_to_channel(channel, "💬 Send 'implement' to apply these changes, or provide additional instructions to modify them.")
        
        # Add AI response to conversation
        ai_message = {"role": "assistant", "content": f"Proposed changes to {len(changes)} files: {', '.join(changes.keys())}"}
        state["messages"].append(ai_message)
    
    # Merge the new changes with existing changes
    existing_changes = state.get("changes", {})
    merged_changes = {**existing_changes, **changes}
    
    # Update the user's conversation state
    conversation_id = get_user_conversation_id(state.get("slack_user"), state.get("slack_channel"))
    if conversation_id in user_conversations:
        user_conversations[conversation_id] = state
    
    return {
        "changes": merged_changes,
        "messages": state["messages"]  # Return updated messages
    }

def commit_push_node(state):
    repo_path = state["repo_path"]
    channel = state.get("slack_channel")
    
    print(f"[DEBUG] Starting commit_push_node with repo_path: {repo_path}")
    send_message_to_channel(channel, f"🔄 Committing and pushing changes...")
    
    try:
        print(f"[DEBUG] Attempting to access git repo at: {repo_path}")
        repo = git.Repo(repo_path)
        
        # Apply changes to files
        changes = state["changes"]
        print(f"[DEBUG] Found {len(changes)} changes to apply")
        
        for file_path, content in changes.items():
            # Use the file_path as relative to repo_path, don't concatenate repo_path again
            if file_path.startswith(repo_path):
                # If file_path already contains the full path, use it as is
                full_path = Path(file_path)
            else:
                # If file_path is relative, join it with repo_path
                full_path = Path(repo_path) / file_path
            
            print(f"[DEBUG] Writing changes to file: {full_path}")
            
            # Ensure parent directory exists
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                full_path.write_text(content)
                print(f"[DEBUG] Successfully wrote to {full_path}")
            except Exception as write_error:
                print(f"[DEBUG] Error writing to {full_path}: {write_error}")
                send_message_to_channel(channel, f"❌ Error writing to {file_path}: {str(write_error)}")
                continue
        
        # Add all changes
        print("[DEBUG] Adding all changes with git add")
        repo.git.add("--all")
        
        # Check if there are any changes to commit
        if not repo.is_dirty() and not repo.untracked_files:
            print("[DEBUG] No changes detected to commit")
            send_message_to_channel(channel, "ℹ️ No changes detected to commit.")
            return state
        
        # Generate commit message using LLM
        print("[DEBUG] Generating commit message")
        diff_output = repo.git.diff("--cached")  # Get staged changes
        system_message = SystemMessage(content="You are a helpful AI software engineer. Please provide a concise, one-line summary of the changes made to the codebase. Focus on the main changes and their purpose.")
        human_message = HumanMessage(content=f"Here are the changes made to the codebase:\n\n{diff_output}\n\nPlease provide a one-line summary for the commit message.")
        
        response = llm.invoke([system_message, human_message])
        commit_message = f"AI: {response.content.strip()}"
        
        # Ensure commit message is not too long
        if len(commit_message) > 100:
            commit_message = commit_message[:97] + "..."
        
        print(f"[DEBUG] Generated commit message: {commit_message}")
        repo.index.commit(commit_message)
        send_message_to_channel(channel, f"✅ Committed changes: {commit_message}")
        
        # Get the origin remote
        origin = repo.remote(name='origin')
        
        # Generate a unique branch name
        base_branch_name = state.get("base_branch_name", "feat/ai-changes-branch-ui-implementations")
        branch_name = base_branch_name
        counter = 1
        
        # Get current branch name to restore later if needed
        current_branch = repo.active_branch.name
        print(f"[DEBUG] Current branch: {current_branch}")
        
        # Try to create a new branch, if it exists, try with a number suffix
        while True:
            try:
                print(f"[DEBUG] Attempting to create branch: {branch_name}")
                # First check if branch exists locally
                if branch_name in repo.heads:
                    print(f"[DEBUG] Branch {branch_name} exists locally, trying next name")
                    branch_name = f"{base_branch_name}-{counter}"
                    counter += 1
                    continue
                
                # Create and checkout new branch
                repo.git.checkout("-b", branch_name)
                print(f"[DEBUG] Successfully created and checked out branch: {branch_name}")
                break
            except git.exc.GitCommandError as e:
                if "already exists" in str(e) or "A branch named" in str(e):
                    branch_name = f"{base_branch_name}-{counter}"
                    counter += 1
                    print(f"[DEBUG] Branch exists, trying: {branch_name}")
                    if counter > 10:  # Prevent infinite loop
                        print(f"[DEBUG] Too many branch attempts, using timestamp")
                        import time
                        branch_name = f"{base_branch_name}-{int(time.time())}"
                        repo.git.checkout("-b", branch_name)
                        break
                else:
                    print(f"[DEBUG] Git checkout error: {e}")
                    raise e
        
        print(f"[DEBUG] Using branch: {branch_name}")
        
        # Configure Git with credentials
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            error_msg = "GITHUB_TOKEN environment variable not set"
            print(f"[DEBUG] Error: {error_msg}")
            send_message_to_channel(channel, f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Set up the remote URL with the token
        repo_url = state["repo_url"]
        if "github.com" in repo_url:
            auth_url = repo_url.replace("https://", f"https://{token}@")
            origin.set_url(auth_url)
            print(f"[DEBUG] Set up authenticated remote URL")
        
        # Try to push the new branch
        push_successful = False
        try:
            print("[DEBUG] Pushing changes to remote...")
            # Push the branch to remote
            origin.push(refspec=f"{branch_name}:{branch_name}")
            print(f"[DEBUG] Successfully pushed branch: {branch_name}")
            send_message_to_channel(channel, f"✅ Successfully pushed branch: {branch_name}")
            push_successful = True
            
        except git.exc.GitCommandError as e:
            print(f"[DEBUG] Push failed with error: {str(e)}")
            
            # Check if it's because the branch already exists on remote
            if "already exists" in str(e) or "non-fast-forward" in str(e) or "failed to push some refs" in str(e):
                print("[DEBUG] Push failed, attempting force push...")
                try:
                    origin.push(refspec=f"{branch_name}:{branch_name}", force=True)
                    print(f"[DEBUG] Successfully force pushed branch: {branch_name}")
                    send_message_to_channel(channel, f"✅ Successfully force pushed branch: {branch_name}")
                    push_successful = True
                    
                except git.exc.GitCommandError as force_error:
                    error_msg = f"Force push failed: {str(force_error)}"
                    print(f"[DEBUG] {error_msg}")
                    send_message_to_channel(channel, f"❌ {error_msg}")
                    
                    # Try with a different branch name
                    print(f"[DEBUG] Trying with timestamp-based branch name...")
                    import time
                    timestamp_branch = f"{base_branch_name}-{int(time.time())}"
                    try:
                        repo.git.checkout("-b", timestamp_branch)
                        origin.push(refspec=f"{timestamp_branch}:{timestamp_branch}")
                        branch_name = timestamp_branch
                        push_successful = True
                        print(f"[DEBUG] Successfully pushed with timestamp branch: {branch_name}")
                        send_message_to_channel(channel, f"✅ Successfully pushed branch: {branch_name}")
                    except Exception as final_error:
                        error_msg = f"Final push attempt failed: {str(final_error)}"
                        print(f"[DEBUG] {error_msg}")
                        send_message_to_channel(channel, f"❌ {error_msg}")
                        return END
            else:
                error_msg = f"Push failed: {str(e)}"
                print(f"[DEBUG] {error_msg}")
                send_message_to_channel(channel, f"❌ {error_msg}")
                return END
        
        if not push_successful:
            error_msg = "Failed to push branch after multiple attempts"
            print(f"[DEBUG] {error_msg}")
            send_message_to_channel(channel, f"❌ {error_msg}")
            return END
        
        # Create a new state with the branch name
        new_state = dict(state)
        new_state["branch"] = branch_name
        print(f"[DEBUG] Successfully completed commit and push. Branch: {branch_name}")
        return new_state
        
    except Exception as e:
        error_msg = f"Error in commit/push process: {str(e)}"
        print(f"[DEBUG] {error_msg}")
        print(f"[DEBUG] Full traceback:")
        traceback.print_exc()
        send_message_to_channel(channel, f"❌ {error_msg}")
        return END

def open_pr_node(state: Dict[str, Any]) -> Dict[str, Any]:
    channel = state.get("slack_channel")
    
    try:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            send_message_to_channel(channel, "❌ GITHUB_TOKEN environment variable not set")
            raise ValueError("GITHUB_TOKEN environment variable not set")
            
        g = Github(token)
        repo_url = state["repo_url"]
        if not repo_url:
            send_message_to_channel(channel, "❌ Repository URL not found")
            raise ValueError("Repository URL not found in state")
            
        # Parse the repository URL to get owner and repo name
        if "github.com" in repo_url:
            parts = repo_url.rstrip(".git").split("/")
            if len(parts) < 2:
                send_message_to_channel(channel, f"❌ Invalid GitHub repository URL format: {repo_url}")
                raise ValueError(f"Invalid GitHub repository URL format: {repo_url}")
            owner = parts[-2]
            repo_name = parts[-1]
            user_repo = f"{owner}/{repo_name}"
            send_message_to_channel(channel, f"🔄 Creating PR for repository: {user_repo}")
        else:
            send_message_to_channel(channel, f"❌ Invalid GitHub repository URL: {repo_url}")
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
            
        try:
            user = g.get_user()
            repo = g.get_repo(user_repo)
            default_branch = repo.default_branch
            
        except Exception as e:
            send_message_to_channel(channel, f"❌ Error accessing repository {user_repo}: {str(e)}")
            return END
        
        # Create the PR
        try:
            # Create PR body
            pr_body = f'''
## AI-Generated Changes

**Requested by:** <@{state.get("slack_user", "unknown")}>
**Instruction:** {state.get("instruction", "No instruction provided")}

### Summary
AI-generated changes to improve the codebase based on the provided instruction.

### Changes Applied
- Applied requested modifications via AI assistant
- Created new branch: `{state.get("branch", "unknown")}`
- Pushed changes to repository

*This PR was created automatically by the AI Slack bot.*
'''
            # Create the PR
            pr = repo.create_pull(
                base=default_branch,
                head=state['branch'],
                title=f"AI: {state.get('instruction', 'Code changes')[:50]}...",
                body=pr_body
            )
            
            send_message_to_channel(channel, f"🎉 PR created successfully!")
            send_message_to_channel(channel, f"🔗 View PR: {pr.html_url}")
            send_message_to_channel(channel, "✅ Task completed! The AI has successfully processed your request.")
            
            return END
            
        except Exception as pr_error:
            send_message_to_channel(channel, f"❌ Error creating PR: {str(pr_error)}")
            return END
            
    except Exception as e:
        send_message_to_channel(channel, f"❌ Error in PR creation process: {str(e)}")
        return END

# Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("clone", clone_node)
builder.add_node("read_repo", read_repo_node)
builder.add_node("apply_change", apply_change_node)
builder.add_node("commit_push", commit_push_node)
builder.add_node("open_pr", open_pr_node)

def should_implement(state):
    """Check if user wants to implement changes"""
    instruction = state.get("instruction", "").lower()
    return instruction == "implement"

def continue_conversation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node to handle conversation continuation"""
    channel = state.get("slack_channel")
    send_message_to_channel(channel, "💭 Ready for your next instruction! You can:")
    send_message_to_channel(channel, "• Give me another instruction to modify the code")
    send_message_to_channel(channel, "• Say 'implement' to apply the current changes")
    send_message_to_channel(channel, "• Say 'reset' to start a new conversation")
    return state

# Add the conversation continuation node
builder.add_node("continue_conversation", continue_conversation_node)

# Set entry point and edges for new conversations
builder.set_entry_point("clone")
builder.add_edge("clone", "read_repo")
builder.add_edge("read_repo", "apply_change")

# Add conditional routing after apply_change
builder.add_conditional_edges(
    "apply_change",
    should_implement,
    {
        True: "commit_push",   # If "implement", go to commit
        False: "continue_conversation"  # Otherwise, wait for more instructions
    }
)

# Continue the implementation flow
builder.add_edge("commit_push", "open_pr")
builder.add_edge("open_pr", END)
builder.add_edge("continue_conversation", END)  # End after showing options

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Global state for tracking active sessions and conversations
active_sessions = {}
user_conversations = {}  # Store conversation history per user

# Hardcoded repository URL - change this to your target repository
REPO_URL = "https://github.com/brianjk17/Bakery-website.git"

def get_user_conversation_id(user: str, channel: str) -> str:
    """Generate a consistent conversation ID for a user in a channel"""
    return f"conversation-{channel}-{user}"

def process_repo_instruction(instruction: str, channel: str, user: str, is_first_message: bool = False):
    """Process a repository instruction in a separate thread"""
    try:
        # Set the base branch name
        BASE_BRANCH_NAME = "feat/ai-slack-bot-change-implementations"
        
        # Get conversation ID for this user
        conversation_id = get_user_conversation_id(user, channel)
        
        # Get existing conversation state or create new one
        if conversation_id in user_conversations and not is_first_message:
            # Continue existing conversation
            send_message_to_channel(channel, f"🔄 Continuing our conversation...")
            
            # Get the existing state
            existing_state = user_conversations[conversation_id].copy()
            
            # Add the new instruction to the conversation
            new_user_message = {"role": "user", "content": instruction}
            existing_state["messages"].append(new_user_message)
            existing_state["instruction"] = instruction
            
            # Configure the graph execution with the existing thread ID
            config = {"configurable": {"thread_id": conversation_id}}
            
            # If user said "implement", we need to run the full implementation flow
            if instruction.lower() == "implement":
                if existing_state.get("changes"):
                    # We have changes to implement - run from commit_push
                    send_message_to_channel(channel, "🚀 Implementing accumulated changes...")
                    print(f"[DEBUG] Starting implementation for user {user}")
                    print(f"[DEBUG] Existing state keys: {existing_state.keys()}")
                    print(f"[DEBUG] Repo path: {existing_state.get('repo_path')}")
                    print(f"[DEBUG] Number of changes: {len(existing_state.get('changes', {}))}")
                    
                    # Set up state for commit and push
                    existing_state["instruction"] = "implement"
                    
                    # Run the commit_push and open_pr nodes directly
                    try:
                        print(f"[DEBUG] Calling commit_push_node...")
                        # Apply changes and commit
                        commit_result = commit_push_node(existing_state)
                        print(f"[DEBUG] commit_push_node returned: {type(commit_result)}")
                        
                        if commit_result != END and commit_result is not None:
                            print(f"[DEBUG] Commit successful, calling open_pr_node...")
                            # If commit was successful, open PR
                            pr_result = open_pr_node(commit_result)
                            print(f"[DEBUG] open_pr_node returned: {type(pr_result)}")
                        else:
                            print(f"[DEBUG] Commit failed or returned END")
                            
                    except Exception as impl_error:
                        error_msg = f"Error during implementation: {str(impl_error)}"
                        print(f"[DEBUG] {error_msg}")
                        print(f"[DEBUG] Implementation error traceback:")
                        traceback.print_exc()
                        send_message_to_channel(channel, f"❌ {error_msg}")
                        
                else:
                    print(f"[DEBUG] No changes found in state for user {user}")
                    send_message_to_channel(channel, "❌ No changes to implement. Please provide an instruction first.")
            else:
                # Continue with new instruction - just run apply_change
                try:
                    result = apply_change_node(existing_state)
                    user_conversations[conversation_id] = result
                    
                    # Show continuation options
                    continue_conversation_node(result)
                    
                except Exception as apply_error:
                    send_message_to_channel(channel, f"❌ Error applying changes: {str(apply_error)}")
            
            # Update stored conversation
            user_conversations[conversation_id] = existing_state
            
        else:
            # Start new conversation
            send_message_to_channel(channel, f"🔄 Starting new conversation...")
            
            # Create initial state
            initial_state = {
                "repo_url": REPO_URL,
                "repo_path": "",
                "instruction": instruction,
                "code": "",
                "modification": "",
                "branch": "",
                "changes": {},
                "messages": [{"role": "user", "content": instruction}],
                "base_branch_name": BASE_BRANCH_NAME,
                "slack_channel": channel,
                "slack_user": user
            }
            
            # Store the conversation
            user_conversations[conversation_id] = initial_state
            
            # Configure the graph execution with the conversation ID
            config = {"configurable": {"thread_id": conversation_id}}
            
            # Start from the beginning
            result = graph.invoke(initial_state, config)
            
            # Update stored conversation with result
            if result:
                user_conversations[conversation_id] = result
        
        # Remove from active sessions when done
        session_key = f"{channel}-{user}"
        if session_key in active_sessions:
            del active_sessions[session_key]
        
    except Exception as e:
        send_message_to_channel(channel, f"❌ Error processing request: {str(e)}")
        print(f"Full error details: {e}")  # For debugging
        # Remove from active sessions on error
        session_key = f"{channel}-{user}"
        if session_key in active_sessions:
            del active_sessions[session_key]

@app.event("app_mention")
def handle_app_mention_events(event, say):
    """Handle app mentions in Slack"""
    user = event['user']
    channel = event['channel']
    text = event['text']
    
    # Remove the bot mention from the text
    cleaned_text = text.split('>', 1)[-1].strip()
    
    # Check if user has an active session
    session_key = f"{channel}-{user}"
    if session_key in active_sessions:
        say(f"Hey <@{user}>, you already have an active session running. Please wait for it to complete before starting a new one.")
        return
    
    # Parse the command - now we only need the instruction
    if not cleaned_text:
        say(f"Hey <@{user}>, please provide an instruction for what you'd like me to change in the repository. Format: `@bot <your instruction>`")
        return
    
    instruction = cleaned_text.lower()
    
    # Check for conversation control commands
    conversation_id = get_user_conversation_id(user, channel)
    
    if instruction == "reset" or instruction == "new conversation":
        # Clear the user's conversation history
        if conversation_id in user_conversations:
            del user_conversations[conversation_id]
        say(f"Hey <@{user}>, I've started a fresh conversation! What would you like me to help you with?")
        return
    
    if instruction == "status" or instruction == "history":
        # Show conversation status
        if conversation_id in user_conversations:
            conv_state = user_conversations[conversation_id]
            message_count = len(conv_state.get("messages", []))
            say(f"Hey <@{user}>, our conversation has {message_count} messages. Last instruction: '{conv_state.get('instruction', 'None')}'")
        else:
            say(f"Hey <@{user}>, we don't have an active conversation yet. Send me an instruction to get started!")
        return
    
    # Regular instruction processing
    instruction = cleaned_text  # Use original case
    is_first_message = conversation_id not in user_conversations
    
    # Mark session as active
    active_sessions[session_key] = {
        "repo_url": REPO_URL,
        "instruction": instruction,
        "started_at": time.time()
    }
    
    if is_first_message:
        say(f"Hey <@{user}>, I'll help you modify the repository! 🚀\n**Repository:** {REPO_URL}\n**Instruction:** {instruction}\n\nStarting the process...")
    else:
        say(f"Hey <@{user}>, continuing our conversation! 💬\n**New instruction:** {instruction}\n\nProcessing...")
    
    # Start processing in a separate thread
    thread = threading.Thread(
        target=process_repo_instruction,
        args=(instruction, channel, user, is_first_message)
    )
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    print("🤖 AI Repo Editor Slack Bot starting...")
    print(f"Bot will modify repository: {REPO_URL}")
    print("Bot will respond to @mentions with instructions")
    print("Format: @bot <instruction>")
    
    handler = SocketModeHandler(app, app_token)
    handler.start()
