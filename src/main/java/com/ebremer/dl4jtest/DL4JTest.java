/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Project/Maven2/JavaApp/src/main/java/${packagePath}/${mainClassName}.java to edit this template
 */

package com.ebremer.dl4jtest;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.eigen.Eigen;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * @author erich
 */
public class DL4JTest {

    public static void main(String[] args) throws IOException {
        
        // Generate Oval 2D point cloud
        int size = 100;
        BufferedImage bi = new BufferedImage(size,size, BufferedImage.TYPE_INT_RGB);
        int offset = 25;
        int rot = 25;
        int major = 50;
        int minor = 25;
        Graphics2D g = bi.createGraphics();
        g.setColor(Color.black);
        g.fillRect(0, 0, size, size);
        g.setColor(Color.BLUE);
        Ellipse2D.Double oval = new Ellipse2D.Double(offset, offset, minor, major);
        g.rotate(Math.toRadians(rot),offset+minor/2,offset+major/2);
        g.fill(oval);
        //g.rotate(Math.toRadians(-15));
        g = bi.createGraphics();
        int count = 0;
        for (int i=0; i<bi.getWidth(); i++) {
            for (int j=0; j<bi.getHeight(); j++) {
                int c = bi.getRGB(i, j) & 0xFF;
                if (c>0) {
                    count++;
                }
            }
        }
        INDArray m = Nd4j.zeros(count,2);
        int h=0;
        for (int i=0; i<bi.getWidth(); i++) {
            for (int j=0; j<bi.getHeight(); j++) {
                int c = bi.getRGB(i, j) & 0xFF;
                if (c>0) {
                   INDArray r = m.getRow(h);
                   r.addi(Nd4j.create(new float[] {i,j}));
                   h++;
                }
            }
        }
        
        // m now contains Test Oval 2D point cloud
       
        int mx = (int) m.getColumn(0).mean(0).getInt(0);
        int my = (int) m.getColumn(1).mean(0).getInt(0);
        
        System.out.println("MC : "+mx+" "+my);

        System.out.println("M    : "+m);
        INDArray z = PCA.pca_factor(m, 2, true);
        System.out.println("FACTOR ====================================");
        System.out.println(z);

        INDArray[] cm = PCA.covarianceMatrix(m);
        System.out.println("Covariance Matrix ====================================");
        System.out.println(cm[0]); 
        INDArray[] aa = PCA.principalComponents(cm[0]);
        INDArray EV = aa[1].rdiv(1.0);
        System.out.println("EV : "+EV);
        double majorlen = 2*Math.sqrt(EV.getDouble(0));
        double minorlen = 2*Math.sqrt(EV.getDouble(1));
        System.out.println("WOW ---> "+majorlen+"  "+minorlen);
        
        System.out.println("Eigenvectors ====================================");
        System.out.println(PCA.principalComponents(cm[0])[0]);
        System.out.println("Eigenvalues ====================================");
        System.out.println(PCA.principalComponents(cm[0])[1]);
        System.out.println("TEST Eigenvalues ====================================");
        INDArray[] result = new INDArray[2];
        INDArray cov = cm[0];
        result[0] = Nd4j.eye(cov.rows());
        result[1] = Eigen.symmetricGeneralizedEigenvalues(result[0], cov, false).rdivi(1.0);
        System.out.println(result[1]);
                
        INDArray ma = z.getColumn(0);
        System.out.println("MAJOR "+ma);
        double a = ma.getDouble(0)*majorlen;
        double b = ma.getDouble(1)*majorlen;
        System.out.println("WOW Major ---> "+a+"  "+b);
        g.setColor(Color.red);
        g.drawLine(mx, my, mx+((int)a), my+((int)b));
        System.out.println("ROT : "+a+" "+b);
        
        INDArray mi = z.getColumn(1);
        System.out.println("MINOR "+mi);        
        a = Math.round(mi.getDouble(0)*minorlen);
        b = Math.round(mi.getDouble(1)*minorlen);
        System.out.println("WOW Minor ---> "+a+"  "+b);
        g.setColor(Color.MAGENTA);
        g.drawLine(mx, my, mx+((int)a), my+((int)b));

        bi.setRGB(mx, my, Color.GREEN.getRGB());
        bi.setRGB(5, 5, Color.WHITE.getRGB());
        ImageIO.write(bi, "png", new File("whoa-"+rot+".png"));
        System.out.println("====================================");
    }
}
