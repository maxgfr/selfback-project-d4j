import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by maxime on 01-Jun-17.
 */
public class ReadFile {

    /** Instance unique non préinitialisée */
    private static ReadFile INSTANCE = null;

    /** Constructeur privé */
    private ReadFile() {
    }

    /** Point d'accès pour l'instance unique du singleton */
    public static synchronized ReadFile getInstance() {
        if (INSTANCE == null)
        { 	INSTANCE = new ReadFile();
        }
        return INSTANCE;
    }

    public void displayFilesForFolder(final File folder) {
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                displayFilesForFolder(fileEntry);
            } else {
                System.out.println(fileEntry.getName());
            }
        }
    }

    public List<String> listFilesPathForFolder (final File folder) {
        List<String> list = new LinkedList<String>();
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesPathForFolder(fileEntry);
            } else {
                list.add(fileEntry.getPath());
            }
        }
        return list;
    }

}
