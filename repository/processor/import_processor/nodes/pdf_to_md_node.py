# 1.内置包
import shutil
import subprocess
import time
import json
from typing import Tuple
from pathlib import Path
# 2.三方包

# 3.自己的包
from repository.processor.import_processor.base import BaseNode, setup_logging
from repository.processor.import_processor.state import ImportGraphState
from repository.processor.import_processor.exceptions import StateFieldError, PdfConversionError

class PdfToMdNode(BaseNode):

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点的处理逻辑入口
        :param state:
        :return:
        """

        # 核心逻辑（接收pdf文件path， 利用mineru解析工具将pdf解析为markdown文件）

        # 1. 获取导入文件的路径以及输出目录
        import_file_path_obj, file_dir_obj = self._validate_state(state)

        # 2. 执行mineru解析（命令： mineru -p input_path -o output_dir" --source=local)
        process_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)
        if process_code != 0:
            raise PdfConversionError(message="MinerU解析PDF失败", node_name=self.name)

        # 3. 获取解析后md_path
        md_path = self._get_md_path(import_file_path_obj, file_dir_obj)

        # 4. 更新state["md_path"]
        state['md_path'] = md_path

        # 5. 返回state
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """

        :param state: 导入图谱节点的状态
        :return: 导入文件路径和输出目录
        Tuple[Path, Path]
        """

        self.log_step("step1", "准备校验和获取解析文件路径和输出目录")

        # 1. 获取解析的文件路径
        import_file_path = state.get("import_file_path", '')

        # 2. 判断文件路径是否存在
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)

        # 3. 标准化解析文件的路径
        import_file_path_obj = Path(import_file_path)

        # 4. 判断是否是一个有效的路径
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str, message="导入文件路径不存在")

        # 5. 获取输出文件目录
        file_dir = state.get("file_dir", '')

        # 6. 判断输出文件目录
        if not file_dir:
            file_dir = import_file_path_obj.parent  # 默认输出目录为导入文件所在目录

        # 8. 标准化输出目录
        file_dir_obj = Path(file_dir)

        # 7. 判断是否是一个有效的目录
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str, message="输出文件目录不存在")

        self.logger.info(f"解析的文件路径: {import_file_path}")
        self.logger.info(f"输出文件目录: {file_dir_obj}")

        return import_file_path_obj, file_dir_obj

    def _execute_mineru_parse(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        """

        :param import_file_path_obj: 解析文件的path路径
        :param file_dir_obj: 解析后的文件输出目录
        :return:  状态[0或者非0]
        0:成功的
        非0:失败的
        """

        # 1. 获取命令的绝对路径
        mineru_executable = shutil.which("mineru")
        if not mineru_executable:
            # 如果依然找不到，说明环境里确实没安装或没配环境变量
            self.logger.error("系统中找不到 mineru 命令，请确认当前环境是否已安装。")
            return -1

        # 2. 定义cmd (将 "mineru" 替换为 mineru_executable)
        cmd = [
            mineru_executable,
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_obj),
            "--source",
            "local"
        ]

        # 2. 子进程执行cmd命令(子进程解析的日志【正常日志和错误日志都要】)
        start_time = time.time()
        proc = subprocess.Popen(args=cmd,
                                stdout=subprocess.PIPE,  # 接收正确日志
                                stderr=subprocess.STDOUT,  # 接收错误日志
                                text=True,  # 输出二进制字节流，输出字符串
                                errors="replace",  # 特殊的字符码替换成?、菱形
                                encoding="utf-8",  # utf-8进行解码
                                bufsize=1  # 实时输出。按行输出遇到\n换行符 就将日志产生出来
                                )

        # 3. 实时打印日志
        for line in proc.stdout:
            self.logger.info(f"MinerU解析产生的日志：{line}")

        # 4. 主线程等待子进程做完（状态码0：成功 反之失败）
        processed_result = proc.wait()

        end_time = time.time()
        if processed_result == 0:
            self.logger.info(f"MinerU解析PDF成功 耗时:{end_time - start_time:.2f}s")
        else:
            self.logger.info("MinerU解析PDF失败")

        return processed_result

    def _get_md_path(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """
        获取解析后md的路径
        :param import_file_path_obj:
        :param file_dir_obj:
        :return:
        md_path= D:\develop\develop\workspace\temp_dir\万用表的使用\hybrid_auto\万用表的使用.md

        Path:吉祥三包：name:全名[文件名字.后缀]   stem【文件名字，没有后缀】    suffix【文件的后缀】
        """

        file_name = import_file_path_obj.stem

        return str(file_dir_obj / file_name / "hybrid_auto" / f"{file_name}.md")

if __name__ == '__main__':
    setup_logging()
    # 1. 构建节点实例
    pdf_to_md_node = PdfToMdNode()

    # 2. 构建该节点状态
    init_state = {
        "import_file_path": r"D:\Project\llm-project\RAG_repository\repository\processor\import_processor\temp\万用表RS-12的使用.pdf",
        "file_dir": r"D:\Project\llm-project\RAG_repository\repository\processor\import_processor\temp"
    }

    # 3. 直接调用process
    result = pdf_to_md_node.process(init_state)

    # 4. 序列化（将对象转成字符串） 反序列化（将字符串转成对象）
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)
