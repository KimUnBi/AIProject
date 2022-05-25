package com.smhrd.service;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

import javax.servlet.ServletException;
import javax.servlet.annotation.MultipartConfig;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.oreilly.servlet.MultipartRequest;
import com.oreilly.servlet.multipart.DefaultFileRenamePolicy;
import com.oreilly.servlet.multipart.FileRenamePolicy;


@WebServlet("/CommonController")
@MultipartConfig(
		fileSizeThreshold = 1024*1024,
		maxFileSize = 1024*1024*50, //50메가
		maxRequestSize = 1024*1024*50*5, //5개까지
		location = "C:\\Users\\cloud\\eclipse-workspace"
		)
public class CommonController extends HttpServlet {
	
	protected void service(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		request.setCharacterEncoding("utf-8");
		// 1. upload 폴더 생성이 안되어 있으면 생성
		String saveDirectory = "C:\\Users\\cloud\\archive\\train\\users";
		System.out.println(saveDirectory);

		File saveDir = new File(saveDirectory);
		if (!saveDir.exists())
			saveDir.mkdirs();

		// 2. 최대크기 설정
		int maxPostSize = 1024 * 1024 * 5; // 5MB  단위 byte

		//3. 인코딩 방식 설정
		String encoding = "UTF-8";

		//4. 파일정책, 파일이름 충동시 덮어씌어짐으로 파일이름 뒤에 인덱스를 붙인다.
	  //a.txt
		//a1.txt 와 같은 형식으로 저장된다.
		FileRenamePolicy policy = new DefaultFileRenamePolicy();
		MultipartRequest mrequest 
		= new MultipartRequest(request //MultipartRequest를 만들기 위한 request
				, saveDirectory //저장 위치
				, maxPostSize //최대크기
				, encoding //인코딩 타입
				, policy); //파일 정책
		
		
//		
		File left_uploadFile = mrequest.getFile("left");
		File front_uploadFile = mrequest.getFile("front");
		File right_uploadFile = mrequest.getFile("right");
		System.out.println(left_uploadFile);
		System.out.println(front_uploadFile);
		System.out.println(right_uploadFile);
		
		response.sendRedirect("html5up_pages/result.html#services");
		
		
//	  //input type="file" 태그의 name속성값을 이용해 파일객체를 생성
//		long uploadFile_length = uploadFile.length();
//		String originalFileName = mrequest.getOriginalFileName("upload"); //기존 이름
//		String filesystemName = mrequest.getFilesystemName("upload"); //기존
	}

}
