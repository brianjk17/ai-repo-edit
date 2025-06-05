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

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = init_chat_model("openai:gpt-4.1")

def replace_value(old: str, new: str) -> str:
    return new

def merge_dicts(old: dict, new: dict) -> dict:
    return {**old, **new}

class State(TypedDict):
    messages: Annotated[List[Dict[str, str]], add_messages]  
    repo_path: Annotated[str, operator.add] 
    repo_url: Annotated[str, operator.add]  
    instruction: Annotated[str, replace_value] 
    code: Annotated[str, operator.add]  
    modification: Annotated[str, operator.add]
    branch: Annotated[str, operator.add]  
    changes: Annotated[dict, merge_dicts] 
    base_branch_name: Annotated[str, replace_value]

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
    # print(f"\n=== Debug: repo_url in clone_node ===")
    # print(f"repo_url: {repo_url}")
    # print(f"repo_url type: {type(repo_url)}")
    # print("=====================================\n")
    
    print(f"Cloning repository: {repo_url}")
    if not repo_url:
        raise ValueError("Repository URL is required")
    
    # Clone the repository and get the path
    repo = clone_repo(repo_url)
    repo_path = str(repo.working_dir)
    print(f"Cloned repository to: {repo_path}")
    
    new_messages = [{"role": "system", "content": f"Cloned repository from {repo_url} to {repo_path}"}]
    
    return {
        "repo_path": repo_path,
        "messages": new_messages
    }

def read_repo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    repo_path = state.get("repo_path")
    if not repo_path or not isinstance(repo_path, str):
        raise ValueError(f"Invalid repository path: {repo_path}")
    
    print(f"Reading repository from path: {repo_path}")
    code = ""
    path = Path(repo_path)
    
    if not path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    for p in path.rglob("*"):
        if p.is_file() and not any(part.startswith('.') for part in p.parts):
            try:
                code += f"\n# File: {p}\n" + p.read_text()
            except Exception as e:
                code += f"\n# Error reading {p}: {e}"
    
    # print(f"Read {len(code)} characters of code")
    
    new_messages = [{"role": "system", "content": f"Read {len(code)} characters of code from repository"}]
    
    return {
        "code": code,
        "messages": new_messages
    }

def get_instruction(state: Dict[str, Any]) -> Dict[str, Any]:
    instruction = input("What would you like to change in the codebase? (Type 'implement' to commit changes): ")
    new_messages = [{"role": "user", "content": instruction}]
    return {
        "instruction": instruction,
        "messages": new_messages
    }

def apply_change_node(state: Dict[str, Any]) -> Dict[str, Any]:
    code = state.get("code", "")
    instruction = state.get("instruction", "")
    
    if instruction.lower() == "implement":
        return state
        
    if not code:
        print("Error: No code content found in state")
        return state
    
    # Get conversation history for context
    messages = state.get("messages", [])
    
    system_messages = [SystemMessage(content="You're a helpful AI software engineer. Please provide the complete modified code for each file that needs to be changed. Format your response as:\n\n# File: path/to/file\n```\nfile contents\n```\n\n# File: path/to/another/file\n```\nfile contents\n```")]
    
    # Convert messages to LangChain message objects
    history_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history_messages.append(AIMessage(content=msg["content"]))
        elif isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
            history_messages.append(msg)
    
    response = llm.invoke(system_messages + history_messages + [
        HumanMessage(content=f"The following codebase is given:\n\n{code}"),
        HumanMessage(content=f"Please apply the following change:\n{instruction}")
    ])
    
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
    
    # Show the proposed changes
    print("\nProposed changes:")
    print("=" * 50)
    if not changes:
        print("No changes were proposed. The LLM might need more context or the instruction might need to be more specific.")
    else:
        for file_path, content in changes.items():
            print(f"\nChanges to {file_path}:")
            print("-" * 30)
            print(content)
    print("=" * 50)
    
    # Ask for confirmation
    # confirmation = input("\nDo you want to apply these changes? (yes/no): ").lower()
    # if confirmation != "yes":
    #     print("Changes rejected. Continuing...")
    #     return state
    
    # Merge the new changes with existing changes
    existing_changes = state.get("changes", {})
    merged_changes = {**existing_changes, **changes}
    
    new_messages = [{"role": "assistant", "content": f"Applied changes to {len(changes)} files"}]
    
    return {
        "changes": merged_changes,
        "messages": new_messages
    }

def commit_push_node(state):
    repo_path = state["repo_path"]
    print(f"Committing changes in: {repo_path}")
    repo = git.Repo(repo_path)
    
    # Apply changes to files
    changes = state["changes"]
    for file_path, content in changes.items():
        full_path = Path(repo_path) / file_path
        print(f"Writing changes to {file_path}")
        full_path.write_text(content)
    
    # Show the changes that will be committed
    print("\nChanges to be committed:")
    print("=" * 50)
    diff_output = repo.git.diff()
    print(diff_output)
    print("=" * 50)
    
    try:
        # Add all changes
        repo.git.add("--all")
        
        # Generate commit message using LLM
        system_message = SystemMessage(content="You are a helpful AI software engineer. Please provide a concise, one-line summary of the changes made to the codebase. Focus on the main changes and their purpose.")
        human_message = HumanMessage(content=f"Here are the changes made to the codebase:\n\n{diff_output}\n\nPlease provide a one-line summary for the commit message.")
        
        response = llm.invoke([system_message, human_message])
        commit_message = f"AI: {response.content.strip()}"
        
        # Ensure commit message is not too long
        if len(commit_message) > 100:
            commit_message = commit_message[:97] + "..."
        
        print(f"Commit message: {commit_message}")
        repo.index.commit(commit_message)
        
        # Get the origin remote
        origin = repo.remote(name='origin')
        
        # Generate a unique branch name
        base_branch_name = state.get("base_branch_name", "feat/ai-changes-branch")  # Get from state or use default
        branch_name = base_branch_name
        counter = 1
        
        # Try to create a new branch, if it exists, try with a number suffix
        while True:
            try:
                repo.git.checkout("-b", branch_name)
                break
            except git.exc.GitCommandError as e:
                if "already exists" in str(e):
                    branch_name = f"{base_branch_name}-{counter}"
                    counter += 1
                else:
                    raise e
        
        print(f"Using branch: {branch_name}")
        
        # Configure Git with credentials
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable not set")

        # Set up the remote URL with the token
        repo_url = state["repo_url"]
        if "github.com" in repo_url:
            auth_url = repo_url.replace("https://", f"https://{token}@")
            origin.set_url(auth_url)
        
        # Try to push with force if needed
        try:
            print("Pushing changes...")
            origin.push(refspec=f"{branch_name}:{branch_name}")
        except git.exc.GitCommandError as e:
            if "failed to push" in str(e):
                print("Normal push failed, attempting force push...")
                try:
                    origin.push(refspec=f"{branch_name}:{branch_name}", force=True)
                    print("Force push successful")
                except git.exc.GitCommandError as force_error:
                    print(f"Force push failed: {str(force_error)}")
                    return END
            else:
                print(f"Push failed: {str(e)}")
                return END
        
        print(f"Successfully pushed branch {branch_name}")
        
        # Create a new state with the branch name
        new_state = dict(state)
        new_state["branch"] = branch_name
        return new_state
        
    except Exception as e:
        print(f"Error in commit/push process: {str(e)}")
        return END

def open_pr_node(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN environment variable not set")
            
        g = Github(token)
        repo_url = state["repo_url"]
        if not repo_url:
            raise ValueError("Repository URL not found in state")
            
        # Parse the repository URL to get owner and repo name
        if "github.com" in repo_url:
            parts = repo_url.rstrip(".git").split("/")
            if len(parts) < 2:
                raise ValueError(f"Invalid GitHub repository URL format: {repo_url}")
            owner = parts[-2]
            repo_name = parts[-1]
            user_repo = f"{owner}/{repo_name}"
            print(f"Creating PR for repository: {user_repo}")
        else:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")
            
        try:
            user = g.get_user()
            print(f"Authenticated as: {user.login}")
            
            repo = g.get_repo(user_repo)
            print(f"Successfully accessed repository: {user_repo}")
            
            # Get repository details
            default_branch = repo.default_branch
            print(f"Using default branch: {default_branch}")
            
        except Exception as e:
            print(f"Error accessing repository {user_repo}: {str(e)}")
            return END
        
        # Create the PR
        try:
            # Create PR body
            pr_body = '''
SUMMARY
AI-generated changes to improve the codebase

CHANGES
- Applied requested modifications
- Created new branch for changes
- Pushed changes to repository
'''
            # Create the PR
            pr = repo.create_pull(
                base=default_branch,
                head=state['branch'],
                title="AI-generated Changes",
                body=pr_body
            )
            print(f"\nPR opened successfully: {pr.html_url}")
            print("\nThank you for using the AI code editor! The session will now end.")
            return END
            
        except Exception as pr_error:
            print(f"Error creating PR: {str(pr_error)}")
            return END
            
    except Exception as e:
        print(f"Error in PR creation process: {str(e)}")
        return END

# Build the graph
builder = StateGraph(State)

# Add nodes
builder.add_node("clone", clone_node)
builder.add_node("read_repo", read_repo_node)
builder.add_node("get_instruction", get_instruction)
builder.add_node("apply_change", apply_change_node)
builder.add_node("commit_push", commit_push_node)
builder.add_node("open_pr", open_pr_node)

# Set entry point and edges
builder.set_entry_point("clone")
builder.add_edge("clone", "read_repo")
builder.add_edge("read_repo", "get_instruction")
builder.add_edge("get_instruction", "apply_change")
builder.add_edge("apply_change", "get_instruction")  # Always loop back to get_instruction

# Add conditional routing for implement
def should_implement(state):
    instruction = state.get("instruction", "")
    return instruction.lower() == "implement"

builder.add_conditional_edges(
    "get_instruction",
    should_implement,
    {
        True: "commit_push",
        False: "apply_change"
    }
)

builder.add_edge("commit_push", "open_pr")
builder.add_edge("open_pr", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # Set your GitHub token
    os.environ["GITHUB_TOKEN"] = os.getenv("GITHUB_TOKEN")
    
    # Set the base branch name for PRs
    BASE_BRANCH_NAME = "feat/ai-add-changes-branch"  # Change this to modify the branch name
    
    # Run the graph with a repository URL
    repo_url = "https://github.com/brianjk17/Bakery-website.git"
    print(f"Initializing with repository URL: {repo_url}")
    
    # Create initial state with empty messages list
    initial_state = {
        "repo_url": repo_url,
        "repo_path": "",
        "instruction": "",
        "code": "",
        "modification": "",
        "branch": "",
        "changes": {},
        "messages": [],  # Initialize empty messages list
        "base_branch_name": BASE_BRANCH_NAME  # Add base branch name to state
    }
    
    # Configure the graph execution with a thread ID
    config = {"configurable": {"thread_id": "1"}}
    graph.invoke(initial_state, config)

# change the text from "The July" to "Brian's bakeshop" 