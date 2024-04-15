from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

# Define question and context
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppeteer."

# Tokenize input (question and text/context)
inputs = tokenizer(question, text, return_tensors="pt")

# Predict answer spans using the model
with torch.no_grad():
    outputs = model(**inputs)

# Retrieve the most likely start and end of answer indices
answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1  # Add 1 to include the end index in the slice

# Convert indices to tokens, then tokens to string to get the answer
answer_tokens = inputs.input_ids[0, answer_start:answer_end]
answer = tokenizer.decode(answer_tokens)

print(answer)



question = "Mức trần có thể tăng giảm trong thời gian triển khai dự án không ?"
text = f"""
ợ trong dự án. Ghi chú: Nếu thiết kế của dự án yêu cầu thì phần trăm giải  ngân cho một hay nhiều hạng mục đạt 100 phần trăm là điều  có thể với điều kiện toàn bộ phần chi phí cho dự án của Ngân  hàng nằm trong khuôn khổ các thông số tài trợ của quốc gia  về chia sẻ chi phí. 41. Đối với các dự án tại các quốc gia chưa thông qua các  thông số tài trợ của quốc gia hoặc chưa được áp dụng Chính  sách Hoạt động 6.00, xem phụ lục I. 42. Ngân hàng có thể yêu cầu bên vay triển khai các hoạt  động nào đó trước khi rút vốn từ khoản vay. Các điều kiện  này được quy định trong hiệp định pháp lý. Các khoản chi  Các hạng mục chi tiêu Tỷ lệ giải ngân Các điều kiện giải ngân 14151415 tiêu bị chi phối bởi các điều kiện như vậy sẽ được tách biệt  rõ ràng với các khoản chi tiêu còn lại trong dự án và được  xếp vào hạng mục chi tiêu riêng biệt, do vậy Ngân hàng có  thể kiểm tra và giám sát điều 
kiện một cách kỹ lưỡng. Sau  đây là một  vài ví dụ về các trường hợp thường có điều kiện  giải ngân đối với một phần của một khoản vay: •  Nếu một dự án có nhiều đơn vị thực hiện, một phần của  khoản vay có thể được quy định dành cho việc thành lập  một đơn vị mới, đơn vị này được cho là không hoạt động  cho đến khi nào dự án được triển khai tốt. •  Nếu dự án yêu cầu việc hoàn thành tài liệu hướng dẫn  về hoạt động và các quy trình hoặc việc thành lập một cơ  quan ra quyết định chuyên phê duyệt các khoản cho vay  nhỏ hoặc các khoản viện trợ nhỏ đối với một phần của  khoản vay, yêu cầu 
những phần việc này phải xong có  thể là điều kiện giải ngân. •  Nếu dự án yêu cầu việc thực hiện một hiệp định pháp lý  phụ đối với một phần của khoản vay, phần này có thể là  điều kiện giải ngân. 43. Những hoạt động có tầm quan trọng hàng đầu đối với  dự án không được xem là các điều kiện giải ngân chẳng hạn  như các hoạt động cần được hoàn thành trước khi hiệp định  vay có hiệu lực. Trong các ví dụ trên đây, việc thành lập một  đơn vị cụ thể hoặc việc hoàn thành sổ tay hoặc việc thực thi  các các thỏa thuận bổ sung, và các điều kiện có liên quan,  không được cản trở việc thực hiện các phần khác của dự án 44. Các khoản thanh
ơng, vì có thể trong trường hợp  đó sẽ có một số các điều kiện phải áp dụng.     108. Mức trần là mức tối đa của khoản tiền vay có thể được  gửi vào tài khoản chuyên dùng trong khi chờ cung cấp cho  Ngân hàng các tài liệu hỗ trợ chứng minh cho việc sử dụng  các khoản tiền đã được tạm ứng. 109. Ngân hàng thường thiết lập mức trần dựa trên các chi  phí dự án đã hoạch định. Ngân hàng cũng xem xét đánh giá  của đội chuẩn bị dự án về khả năng của bên vay nhằm đảm  bảo việc sử dụng có hiệu quả các tài khoản chuyên dùng.  Ngân hàng có thể thiết lập mức trần (a) là một khoản cố định,  hoặc (b) là khoản có thể được điều chỉnh tuỳ vào từng thời  điểm trong quá trình thực hiện dự án dựa trên dự báo định  kỳ nhu cầu về dòng tiền mặt (xem tiểu mục 6.1 của Hướng  Mức trần  34353435 GIẢI NGÂN CHO BÊN VAY dẫn Giải ngân). Bên vay có thể yêu cầu các khoản tạm ứng  khi cần thiết cho việc thực hiện dự án chừng nào tổng khoản  tiền chưa có tài liệu hỗ trợ chứng minh không vượt quá mức  trần. Khi đạt đuợc mức trần, bên vay phải báo cáo về việc  những khoản tiền được tạm ứng trước đó đã được sử dụng  như thế nào trước khi Ngân hàng giải ngân bất kỳ khoản  tạm ứng bổ sung nào hoặc cung cấp bằng chứng về việc yêu  cầu ngay tiền mặt làm tăng mức trần. 110. Mức trần cố định. Mức trần là một khoản tiền cố định có  thể được phân bổ khi các chi phí được dự định sẽ phát sinh  thậm chí trong suốt thời hạn của dự án. Mức trần thường  được qui định cho toàn bộ thời hạn của dự án và có thể 
là  trung bình quân của các chi phí dự án đã được hoạch định. 111. Phụ thuộc vào khả năng và việc thực hiện của bên vay,  mức trần này có thể được thay đổi. Ví dụ, mức trần có thể  tăng khi bên vay chứng minh khả năng quản lý tài khoản  chuyên dùng của mình trong phạm vi mức trần; một cách  tương ứng, mức trần có thể bị giảm xuống nếu thực hiện yếu  kém hoặc chậm chạp trong việc thực hiện dự án.  112. Cơ sở để thiết lập mức trần cũng có thể được thay đổi  từ một khoản cố định thành một khoản dựa trên các dự báo  định kỳ. Ví dụ, khoản này có thể được phân bổ cho các bên  vay quan tâm đến việc s�ơ sở để thiết lập mức trần cũng có thể được thay đổi  từ một khoản cố định thành một khoản dựa trên các dự báo  định kỳ. Ví dụ, khoản này có thể được phân bổ cho các bên  vay quan tâm đến việc sử dụng các báo cáo tài chính giữa kỳ  chưa kiểm toán (bao gồm cả các dự báo định kỳ) để hỗ trợ  cho việc giải ngân. Để biểt thêm thông tin liên quan đến quá  trình thay đổi mức trần, xem mục “Các vấn đề Giải ngân  trong Quá trình Thực hiện Dự án” tại Chương VII. 113. Mức trần dựa trên Dự báo Định kỳ. Có thể phân bổ  mức trần dựa trên các dự báo định kỳ khi dự kiến những  khoản chi phí của dự án 
sẽ phát sinh khác nhau trong suốt  thời hạn dự án nhằm đáp ứng các nhu cầu thực hiện (chẳng  hạn như, những dao động theo mùa, việc phân chia giai   đoạn các cấu thành dự án). Trong trường hợp này, mức trần  có thể dựa trên (a) những dự báo của bên vay như được trình  bày trong các báo cáo tài chính giữa kỳ chưa kiểm toán; hoặc  (b) những ước tính của đội chuẩn bị dự án về các chi phí dự  án hoạch định, có thể phát sinh, ví dụ như, từ kế hoạch mua  sắm hoặc ngân sách hàng năm. Khi mức trần được dựa trên  các dự báo định kỳ, khoản tiền thực sự của mức trần có thể  thay đổi theo từng giai đoạn để phản ánh những thay đổi  trong nhu cầu về dòng tiền mặt. Ngân hàng sẽ đánh giá sự  hợp lý của các dự báo và có thể điều chỉnh khoản tiền mà  Ngân hàng sẵn sàng tạm ứng nếu không thoả mãn rằng dự  báo đó được chứng minh bởi các chi phí dự án hoạch định  (xem tiểu mục 6.4 của Hướng dẫn Giải ngân). 114. Trong trường hợp Ngân hàng quyết định rằng, dựa trên  kinh nghiệm về dự án và việc thực hiện của bên vay, các dự  báo định kỳ không hiệu quả đối với việc đặt ra mức trần,  thì cơ sở thiết lập mức trần có thể được thay đổi thành một  khoản cố định (xem mục “Các Vấn đề Giải ngân trong Quá  trình Thực hiện Dự án” tại chương VII). SỔ TAY GIẢI NGÂN DÀNH CHO KHÁCH HÀNG CỦA NGÂN HÀNG THẾ GIỚI 36373637 115. Yêu cầu tạm ứng thường không cần tài liệu hỗ trợ vào  thời điểm yêu cầu nếu khoản yêu cầu nằm trong mức trần  đã được thoả thuận.  Nếu việc thanh toán khoản tiền có thể  dẫn 
đến việc vượt quá mức trần
"""

# Tokenize input (question and text/context)
inputs = tokenizer(question, text, return_tensors="pt")

# Predict answer spans using the model
with torch.no_grad():
    outputs = model(**inputs)

# Retrieve the most likely start and end of answer indices
answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
answer_start = torch.argmax(answer_start_scores)
answer_end = torch.argmax(answer_end_scores) + 1  # Add 1 to include the end index in the slice

# Convert indices to tokens, then tokens to string to get the answer
answer_tokens = inputs.input_ids[0, answer_start:answer_end]
answer = tokenizer.decode(answer_tokens)

print(answer)

# question = "Tạm ứng dư thừa/khoản dư thừa là gì ?"
# text = f"""
# thanh toán.  •  Tạm ứng: là khoản thanh toán cho bên vay các khoản chi phí của dự án đã được  dự trù trước. Bên vay, theo định kỳ nộp các chứng từ cho thấy rằng các chi phí đã  phát sinh và được chi trả từ tiền được tạm ứng Giải ngân cho các bên thứ ba •      Thanh toán trực tiếp: Thanh toán cho bên thứ ba (như nhà thầu, nhà cung  cấp, tư vấn) đối với chi phí cho các khoản chi tiêu của dự án. Bên vay cung cấp  chứng từ chứng minh rằng các khoản chi tiêu như vậy đã được thực hiện vào thời  điểm có yêu cầu thanh toán cho bên thứ ba.  • Cam kết đặc biệt: Thanh toán cho một tổ chức tài chính đối với chi phí cho các  khoản chi tiêu của dự án được thực hiện theo một cam kết đặc biệt. Một cam kết  đặc biệt là một cam kết không hủy ngang 
# với sự tham gia của Ngân hàng bằng  văn bản cam kết thanh toán các khoản đó kể cả khi bị đình chỉ hoặc hủy vốn  nào sau đó. Tổ chức tài chính đưa ra xác nhận rằng các khoản chi tiêu như vậy đã  được thực hiện vào thời điểm được yêu cầu thanh toán. 18191819 kế hoạch mua sắm đấu thầu, và kinh nghiệm giải ngân của  bên vay với Ngân hàng cũng được đưa vào xem xét cùng với  các vấn đề sau: •  bản chất và số lượng các khoản chi tiêu dự kiến của dự án  và các yêu cầu của các nhà cung cấp mà có thể thấy trước  được, •  kế hoạch tài chính tổng thể và khả năng tài trợ đối ứng  kịp thời của bên vay, •  mức độ tập trung hóa phân tán của người hưởng lợi của  dự án, •  khả năng tài trợ trước của bên vay cho các khoản chi  tiêu, •  yêu cầu định 
# kỳ về tiền mặt của dự án, •  các bước chuẩn bị báo cáo tài chính. Chứng từ Hỗ trợ 57. Ngân hàng yêu cầu chứng từ hỗ trợ mà các chứng từ  này cung cấp được bằng chứng cho thấy việc rút vốn từ tài  khoản vay đã được hoặc đang được dành cho các khoản chi  tiêu hợp lệ, như được định rõ trong hiệp định pháp lý và  phần 4 của Hướng dẫn Giải ngân. Tùy theo phương pháp  giải ngân được sử dụng, bên vay có thể cung cấp chứng từ  này cùng với lúc nộp 
# đơn xin rút vốn hoặc vào một ngày  sau đó (xem phần về “Việc sử dụng Phương pháp tạm ứng”  trong chương V). 58. Chứng từ hỗ trợ có thể dưới dạng (a) copy của chứng từ  gốc chứng minh rằng thanh toán đã được thực hiện hoặc  đến hạn phải trả cho các chi tiêu hợp lệ (ví dụ hóa đơn
# biệt).  Tỉ lệ giải ngân: là tỉ lệ phần trăm các chi tiêu  hợp lệ được cấp vốn theo một dự án.  Quyết toán Chứng từ: là thuật ngữ chung  được sử dụng để đưa ra bằng chứng hỗ trợ cho  một quyết định hay hành động có được thực  hiện hay không. Trong ngữ cảnh kế toán, thông  thường, quyết toán chứng từ cung cấp các tài  liệu hỗ trợ chẳng hạn như các chứng từ, hoá đơn  của bên bán, chấp nhận thanh toán, thông báo  vận chuyển, vv.) hoặc báo cáo chi phí tóm tắt để  hỗ trợ cho việc thanh toán, hoàn trả, nhập bút  toán kế toán, hoặc sự kiện kế toán khác (xem  định nghĩa về chứng từ, tài liệu hỗ trợ, và báo  cáo tóm tắt). 62636263 PHỤ CHƯƠNG A - ĐỊNH NGHĨA CÁC THUẬT NGỮ  Ngày hiệu lực: là ngày mà Ngân hàng thông  báo cho bên vay Ngân hàng chấp nhận bằng  chứng yêu cầu để chứng minh các điều kiện hiệu  quả theo hiệp định pháp lý đã được đáp ứng, và  là ngày hiệp định pháp lý có hiệu lực. Việc rút  vốn có thể thực hiện từ tài khoản vay kể từ ngày  này, chẳng hạn như, bẳt đầu giải ngân.  Chi tiêu hợp lệ: là chi tiêu hợp lệ, theo hiệp  định pháp lý, sẽ được chi trả từ tiền của khoản  vay.    Tạm ứng dư thừa/khoản dư thừa: là khoản  tiền được gửi vào tài khoản chuyên dùng mà,  trong thời gian tới,  không cần thiết cho việc chi  trả các khoản thanh toán những chi tiêu hợp lệ.    Hạng mục chi tiêu: là hạng mục chi tiêu hợp  lệ 
# có thể được tài trợ từ tiền của khoản vay.    Tổ chức tài chính: là ngân hàng thương mại,  ngân hàng trung ương, hoặc các tổ chức khác  đáp ứng được các tiêu trí trong Hướng dẫn Giải  ngân để nắm giữ tài khoản chuyên dùng.  Khoản tự làm: là các công trình dân dụng  được  thực  hiện  bởi  cơ  quan  chính  phủ  địa  phương của bên vay sử dụng lực lượng lao động  của riêng họ.  Phí ban đầu: là khoản phí IBRD tính phí các  bên vay về khoản vay, khoản phí này phải thanh  toán vào ngày hiệu lực của khoản vay, và theo  quyền tự quyết của bên vay, cũng như được qui  định cụ thể trong hiệp định pháp lý, có thể được  chi trả từ tiền vay.  Điều kiện Chung đối với các khoản Tín  dụng và Tài trợ: là những điều khoản và điều  kiện chuẩn được đưa ra để áp dụng cho các hiệp  định tài trợ giữa bên nhận tài trợ và IDA, theo các  điều khoản của hiệp định nà
# t toán (bằng chữ) 16. Nếu chứng từ chung cho nhiều khoản vay (theo mục 2 ở trên), điền số tiền phân bổ cho từng khoản tài trợ      Số tham chiếu khoản vay/tài trợ/viện trợ             Số tiền                        Số tham chiếu khoản vay/tài trợ/viện trợ                                Số tiền 14. Hạng mục và số tham chiếu hợp đồng - nếu chi tiêu liên quan  tới nhiều hạng mục hay tham chiếu hợp đồng thì để trống mục 14a và  14b và bổ sung thông tin này vào trong chứng từ kèm theo 14a. Tham chiếu hạng mục   14b. Tham chiếu hợp đồng D. Xác nhận và ký tên Người ký tên dưới đây xác nhận và đảm bảo những điều sau: A. Nếu người ký tên yêu cầu Tạm ứng vào tài khoản chuyên dùng: (1) số tiền yêu cầu tương ứng với chi phí dự kiến của dự án 
# đã trình cho Ngân hàng thế giới; và (2) chứng từ của khoản ứng trước này (sẽ)  được chuyển cho Ngân hàng thế giới theo kỳ báo cáo nêu trong các hiệp định  pháp lý liên quan hoặc thư giải ngân của dự án. Nếu người ký tên dưới đây quyết toán các thanh toán từ tài khoản chuyên  dùng: (a) chi tiêu nêu trong đơn hợp lệ được tài trợ từ khoản vay/tài trợ/viện trợ theo các điều khoản của (các) hiệp định có tính pháp lý, và (b) người ký tên dưới đây trước đây chưa và sẽ không có ý định  tìm nguồn tài chính khác để sử dụng cho những khoản chi tiêu này. B. Nếu người ký tên dưới đây yêu cầu hoàn vốn hay thanh toán trực tiếp: (1) những chi tiêu nêu trong đơn thuộc đối tượng cấp kinh phí từ các đợt giải ngân của khoản vay/tài trợ/viện trợ theo 
# các điều khoản  của (các) văn bản pháp lý liên quan; và (b) người ký tên dưới đây chưa bao giờ và sẽ không có ý định tìm nguồn tài chính khác để sử 
# dụng cho những khoản chi này. C. Nếu người ký tên dưới đây yêu cầu giải ngân đợt cấp vốn trong phạm vi khoản vay chính sách phát triển/tài trợ/viện 
# trợ: (1) số tiền tương đương với số tiền ký quỹ  sẽ được chịu trách nhiệm kế toán theo các  điều khoản của (các) văn bản pháp lý liên quan, và (2) sẽ không sử dụng nguồn vốn của khoản vay/tài trợ/viện trợ để dùng cho chi tiêu không hợp lệ nêu trong các văn bản pháp lý liên quan.Người ký tên  dưới đây sẽ lưu giữ toàn bộ chứng từ các chi tiêu thuộc Đơn xin rút vốn này để cung cấp cho bên kiểm toán và cán bộ kiểm tra của Ngân hàng thế giới. 17. Tê
# """

# inputs = tokenizer(question, text, return_tensors="pt")
# with torch.no_grad():
#     outputs = model(**inputs)

# print(outputs)



