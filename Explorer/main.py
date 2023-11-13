import click

@click.group()
def main() -> None:
    pass

@click.command()
def hello_world() -> None:
    print("Testing - Hello world!")

main.add_command(hello_world)

if __name__ == "__main__":
    main()
