import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

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

    public void listFilesForFolder(final File folder) {
        for (final File fileEntry : folder.listFiles()) {
            if (fileEntry.isDirectory()) {
                listFilesForFolder(fileEntry);
            } else {
                System.out.println(fileEntry.getName());
            }
        }
    }

    public void readAllFileFromRes () {


        final File folder = new File("/home/you/Desktop");
        listFilesForFolder(folder);
        try (Stream<Path> paths = Files.walk(Paths.get("/home/you/Desktop"))) {
            paths
                    .filter(Files::isRegularFile)
                    .forEach(System.out::println);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
