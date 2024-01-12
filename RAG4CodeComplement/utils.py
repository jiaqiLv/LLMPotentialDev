def file2str(file_path):
    """
    将文件内容转换为字符串形式
    """
    with open(file_path,'r',encoding='utf-8') as file:
        file_content = file.read()
    file_content_with_newline = file_content.replace('\n','\\n')
    return file_content_with_newline