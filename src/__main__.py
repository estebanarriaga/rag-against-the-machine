from fire import Fire
from .rag import RAG

def main() -> None:
    '''
    Main function to run the RAG application.
    '''

    try:
        Fire(RAG)
    except KeyboardInterrupt:
        print('Exiting...')


if __name__ == '__main__':
    main()
