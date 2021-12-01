import React, { Component } from "react";
import { Spinner } from "react-bootstrap";

class UploadFiles extends Component {
  constructor(props) {
    super(props);
    this.state = {
      loading: false,
      selectedFile: null,
      errorMsg: "",
      successMsg: "",
    };
    this.onSubmit = this.onSubmit.bind(this);
    this.onChangeHandler = this.onChangeHandler.bind(this);
    this.renderSpinnerOrMessage = this.renderSpinnerOrMessage.bind(this);
  }

  onSubmit() {
    const { endpoint, errorMsg, successMsg, responseCallback } = this.props;
    this.setState({ loading: true });
    const data = new FormData();
    for (let i = 0; i < this.state.selectedFile.length; i++) {
      data.append(
        "file",
        this.state.selectedFile[i],
        this.state.selectedFile[i].name
      );
      console.log(this.state.selectedFile[i]);
    }
    fetch(endpoint, {
      method: "POST",
      body: data,
    })
      .then((res) => {
        return res.json();
      })
      .then((data) => {
        console.log(data);
        this.setState({
          loading: false,
          successMsg: successMsg ? successMsg : "Success!",
          errorMsg: "",
        });
        responseCallback(data);
      })
      .catch((error) => {
        this.setState({
          loading: false,
          successMsg: "",
          errorMsg: errorMsg ? errorMsg : "There was an error!",
        });
        console.log(error);
      });
  }

  onChangeHandler(event) {
    this.setState({
      selectedFile: event.target.files,
    });
  }

  renderSpinnerOrMessage() {
    const { loading, errorMsg, successMsg } = this.state;
    return loading ? (
      <Spinner animation="border" size="sm" />
    ) : errorMsg ? (
      <span style={{ color: "red" }}>{errorMsg}</span>
    ) : successMsg ? (
      <span style={{ color: "green" }}>{successMsg}</span>
    ) : null;
  }

  render() {
    const { title, onSubmit, buttonText } = this.props;
    return (
      <div
        style={{
          width: "80%",
          padding: 25,
          margin: "0px auto",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            width: "50%",
            margin: "0px auto",
          }}
        >
          {title ? <h1>{title}</h1> : null}
          <input
            type="file"
            name="file"
            multiple
            onChange={(e) => this.onChangeHandler(e)}
            style={{ marginBottom: 10 }}
          />
          {this.renderSpinnerOrMessage()}
          <button
            type="button"
            className="btn btn-primary btn-block"
            style={{ marginTop: 10 }}
            onClick={() => (onSubmit ? onSubmit : this.onSubmit())}
          >
            {buttonText}
          </button>
        </div>
      </div>
    );
  }
}

export { UploadFiles };
