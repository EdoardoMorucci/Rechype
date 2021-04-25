package it.unipi.dii.inginf.lsmdb.rechype;

import it.unipi.dii.inginf.lsmdb.rechype.user.*;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class App extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception{
        Parent root = FXMLLoader.load(getClass().getResource("it/unipi/dii/inginf/lsmdb/rechype/gui/landing.fxml"));
        primaryStage.setTitle("Hello World");
        primaryStage.setScene(new Scene(root));
        primaryStage.show();
    }

    private static final UserServiceFactory factory=UserServiceFactory.create();

    public static void main(String[] args){

        launch(args);

    }
}
