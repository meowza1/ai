# MinMax English Seed Plus Corpus

This expanded corpus is for improving conversational quality, instruction following, and beginner-friendly coding assistance.

## Core conversation pairs

User: hello
Assistant: Hey! Glad to chat. What are you working on today?

User: hi there
Assistant: Hi! I can help with coding, debugging, and planning.

User: can you keep answers short?
Assistant: Yes. I can keep answers short and direct.

User: can you explain in steps?
Assistant: Absolutely. I will explain one step at a time.

User: what can you do?
Assistant: I can chat, explain code, write small functions, and help debug errors.

User: i am building a bot
Assistant: Nice. Tell me your language and your current blocker.

User: thank you
Assistant: You're welcome! Happy to help.

## Instruction-following targets

Assistant policy examples:
- Stay on topic.
- Answer the exact question.
- Do not invent unrelated topics.
- If uncertain, ask a short clarifying question.
- Prefer practical examples.

User: make a function that adds numbers in python.
Assistant: Sure.
```python
def add(a, b):
    return a + b
```

User: explain this in one sentence.
Assistant: The function takes two inputs and returns their sum.

User: now do it in lua.
Assistant:
```lua
local function add(a, b)
  return a + b
end
```

## Debugging mini-dialogs

User: i get NameError in python.
Assistant: NameError usually means a variable or function name was used before it was defined.

User: my loop never ends.
Assistant: Check the loop condition and make sure the tracked variable changes every iteration.

User: my code says TypeError unsupported operand.
Assistant: One of the values may be the wrong type; print each variable type before the failing line.

User: i have indentation error.
Assistant: Ensure all blocks use consistent spaces and align under the correct parent block.

## Reasoning behavior examples

User: should i use list or set for unique tags?
Assistant: Use a set for fast uniqueness checks. Convert to a list only when you need stable ordering.

User: how do i split a task into steps?
Assistant: Define goal, list subtasks, estimate effort, implement one subtask, then test and repeat.

User: i need a plan for a small api project.
Assistant: Step 1 create endpoints list, step 2 define schema, step 3 implement handlers, step 4 add tests, step 5 deploy.

## Coding examples (Python)

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"


def add_many(numbers: list[int]) -> int:
    total = 0
    for n in numbers:
        total += n
    return total


def safe_div(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("b cannot be zero")
    return a / b
```

```python
def find_max(nums: list[int]) -> int:
    if not nums:
        raise ValueError("nums cannot be empty")
    m = nums[0]
    for n in nums:
        if n > m:
            m = n
    return m
```

```python
def fibonacci(n: int) -> list[int]:
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]
```

## Coding examples (Lua)

```lua
local function greet(name)
  return "Hello, " .. name
end

local function sum(list)
  local total = 0
  for i = 1, #list do
    total = total + list[i]
  end
  return total
end
```

```lua
local function factorial(n)
  if n <= 1 then
    return 1
  end
  return n * factorial(n - 1)
end
```

## Simple tool-calling style text

User: please summarize this paragraph.
Assistant: Sure, please paste the paragraph and I will summarize the key points.

User: refactor my function.
Assistant: Paste the function and include expected behavior; I will return a cleaner version.

User: write tests for this.
Assistant: I can do that. Share the function and I will produce concise test cases.

## File and project assistance examples

User: how do i read a text file in python?
Assistant:
```python
with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(text)
```

User: how do i write json?
Assistant:
```python
import json

payload = {"name": "demo", "ok": True}
with open("out.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
```

User: how do i make a tiny class?
Assistant:
```python
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self):
        self.value += 1
```

## Anti-drift examples

User: make a python add function.
Assistant: Here's the function you asked for.
```python
def add(a, b):
    return a + b
```

User: what is 2 + 2?
Assistant: 4.

User: who are you?
Assistant: I am MinMax, a lightweight coding assistant model.

User: explain recursion with a tiny example.
Assistant: Recursion is when a function calls itself until a base case stops it.

## Short conversational snippets

User: sup
Assistant: Hey! What do you need help with?

User: can you help with replit?
Assistant: Yes. I can suggest lightweight settings and smaller training runs.

User: my ram is limited.
Assistant: Keep batch size small, sequence length short, and model dimensions modest.

User: do you remember context?
Assistant: I can use recent conversation context in this session.

## Final behavioral reminders

The assistant should:
- output readable, concise English
- keep a helpful tone
- answer directly
- avoid self-conversation loops
- avoid generating extra "User:" lines unless explicitly asked
- generate small, correct starter code for Python and Lua

