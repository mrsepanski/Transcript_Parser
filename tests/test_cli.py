from transcript_parser.cli import greet

def test_greet():
    assert greet("World") == "Hello, World!"
