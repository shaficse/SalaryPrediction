l = 0
def test_input():
        try:
            l= 3
        except Exception as e:
            assert False, f"Model failed to take expected input: {e}"
    
def test_output_shape():    
    try:
        print(f"Love bird {l}")
    except Exception as e:
        assert False, f"Output shape test failed: {e}"

if __name__ == '__main__':
    # If the script is run directly, execute your app or other logic
    test_input()
    test_output_shape()