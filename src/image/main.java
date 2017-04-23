package image;
import java.awt.Color;
import java.awt.image.*;

import javax.imageio.*;

import java.io.*;
import java.util.Scanner;
public class main {

	public static void main(String[] args) throws IOException 
	{
		FileWriter fw = new FileWriter("rgb.txt");
		BufferedWriter bw = new BufferedWriter(fw);
		String rgb = "";
	    BufferedImage img = ImageIO.read(new File("dice.png"));
			
	    	System.out.println(img.getHeight() +" "+ img.getWidth());
	    	Scanner s = new Scanner(System.in);
	    	s.nextLine();
		    double[][] C=new double[img.getHeight()][img.getWidth()];
		    for(int i=0;i<img.getHeight();i++) 
		    {
		        for(int j=0;j<img.getWidth();j++)
		        {
		        Color red = new Color(img.getRGB(j, i));
		        rgb = rgb + red.getRed()+ " ";
		      //  System.out.print(red.getRed() + " ");
		        }
		      //  System.out.println();
		        rgb = rgb + '\n';
		    }  
		   bw.write(rgb);
}
}
