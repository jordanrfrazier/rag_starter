def query_loop(chain):
    while True:
        query = input("Enter a question:\n")
        response = chain.run(input=query)
        print(f"Response:\n{response}")
