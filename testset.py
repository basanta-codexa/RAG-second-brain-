# Each item: a question, and the (source, page) pairs that hold the answer.
# page=None means "any page of that source is fine" and is reported as a
# *lenient* match by evaluate.py -- prefer page-specific answers where possible.

# Sources referenced below are expected under --folder (default: data). Tests
# whose source is missing are skipped and reported, not silently dropped.

TESTS = [
    # System Design Interview Playbook
    {
        "q": "What are the seven stages of the 45-minute interview flow and the time for each?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "Which quantities should be estimated during capacity planning?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "How should an engineer justify choosing a particular technology?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What questions should be asked to clarify the scope of a system design problem?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "How do you calculate average and peak QPS using the playbook's example?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "Which invariants should be defined before choosing a database?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What information should an API or event contract include?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What is the baseline architecture described in the system design playbook?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What are the trade-offs an engineer should consider when choosing a data store?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "How should a system handle duplicate requests or events?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What strategies are recommended for handling a queue backlog?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What should a system do when its cache becomes unavailable?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "Which metrics should be monitored to determine whether a system is healthy?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What are the six categories in the system design self-scoring rubric?",
        "answers": [("interview-playbook.md", 1)]
    },
    {
        "q": "What artifacts should be produced for each system design practice problem?",
        "answers": [("interview-playbook.md", 1)]
    },

    # Machine Learning notes
    {
        "q": "What is a generative learning algorithm?",
        "answers": [("main_notes.pdf", 29)]
    },
    {
        "q": "What is binary classification?",
        "answers": [("main_notes.pdf", 15)]
    },
    {
        "q": "What is the learning rate in gradient descent?",
        "answers": [("main_notes.pdf", 4)]
    },
    {
        "q": "What is supervised learning?",
        "answers": [("main_notes.pdf", 1), ("main_notes.pdf", 2)]
    },
    {
        "q": "What is the difference between regression and classification?",
        "answers": [("main_notes.pdf", 2)]
    },
    {
        "q": "What is a training set, and what is a training example?",
        "answers": [("main_notes.pdf", 1)]
    },
    {
        "q": "What is a hypothesis in supervised learning?",
        "answers": [("main_notes.pdf", 2)]
    },
    {
        "q": "What is the cost function used in linear regression?",
        "answers": [("main_notes.pdf", 4)]
    },
    {
        "q": "How does gradient descent update model parameters?",
        "answers": [("main_notes.pdf", 4), ("main_notes.pdf", 5)]
    },
    {
        "q": "What is the difference between batch gradient descent and stochastic gradient descent?",
        "answers": [("main_notes.pdf", 5), ("main_notes.pdf", 7)]
    },

    # System Design Handbook
    {
        "q": "What are the basic principles of system design?",
        "answers": [("system-design-notes.md", 2)]
    },
    {
        "q": "Why should a system design problem be broken into smaller modules?",
        "answers": [("system-design-notes.md", 2)]
    },
    {
        "q": "What is load balancing, and why is it used in distributed systems?",
        "answers": [("system-design-notes.md", 4), ("system-design-notes.md", 5)]
    },
    {
        "q": "What are the main types of load distribution described in the notes?",
        "answers": [("system-design-notes.md", 4)]
    },
    {
        "q": "What responsibilities does a software load balancer have?",
        "answers": [("system-design-notes.md", 6)]
    },
    {
        "q": "What is the difference between SQL and NoSQL databases?",
        "answers": [("system-design-notes.md", 7)]
    },
    {
        "q": "When is an ACID-compliant database a suitable choice?",
        "answers": [("system-design-notes.md", 10)]
    },
    {
        "q": "What are the advantages of NoSQL databases described in the notes?",
        "answers": [("system-design-notes.md", 11)]
    },
    {
        "q": "What is the CAP theorem, and which properties does it discuss?",
        "answers": [("system-design-notes.md", 12), ("system-design-notes.md", 13)]
    },
    {
        "q": "Why can a distributed data store not guarantee consistency, availability, and partition tolerance simultaneously?",
        "answers": [("system-design-notes.md", 13)]
    },
    {
        "q": "How do redundancy and replication improve system reliability?",
        "answers": [("system-design-notes.md", 14)]
    },
    {
        "q": "What is caching, and what principle makes it useful?",
        "answers": [("system-design-notes.md", 15)]
    },

    # The Bhagavad-Gita text
    {
        "q": "Why does Arjuna hesitate before the battle begins?",
        "answers": [("text_data.txt", None)]
    },
    {
        "q": "What comparison does Krishna make between the soul changing bodies and changing worn-out clothes?",
        "answers": [("text_data.txt", None)]
    },
    {
        "q": "What does Krishna teach about performing one's duty without attachment to the results?",
        "answers": [("text_data.txt", None)]
    },
    {
        "q": "How does Krishna describe the nature of the soul?",
        "answers": [("text_data.txt", None)]
    },
    {
        "q": "What does the text teach about controlling the mind?",
        "answers": [("text_data.txt", None)]
    },
    {
        "q": "How are desire and anger connected to the loss of wisdom?",
        "answers": [("text_data.txt", None)]
    },
    {
        "q": "What does Krishna teach about remaining balanced in pleasure and pain?",
        "answers": [("text_data.txt", None)]
    },
    {
        "q": "What is the central subject of the chapter titled 'Virtue in Work'?",
        "answers": [("text_data.txt", None)]
    },
]
