import os

def export_summary(lines, topic_id, method, output_path):
    """
    Purpose: To print out a summary to a file
    Input: [lines]: a list of sentences in a summary (1D list);
           [topic_id]: The first 6 characters and digits of the directory containing the documents used for 
                       generating the summarization (str);
            - Example input: The topic id of "D1001A-A" would be "D1001A" (see /outputs/devtest/)
           [method]: The method for summarization -- "1" or "2" (str; not a digit);
           [output_path]: the output path that the input summary should be placed in (str)
    Output: This function generates a file in a designated directory. Nothing is returned
    """
    content = "\n".join(lines)
    output_filename = topic_id[:5] + "-A.M.100." + topic_id[-1] + "." + method

    # check if directory exists, if not, create one
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # output summary
    with open(os.path.join(output_path, output_filename), "w", encoding = "utf8") as f:
        f.write(content)

    return

if __name__ == "__main__":
    lines = ["this is the first line", "this is the second", "just for testing"]
    topic_id = "D1001A"
    method = "1"
    output_path = "../outputs/D3/"
    export_summary(lines, topic_id, method, output_path)